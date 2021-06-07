import torch
import faiss
import math
import numpy as np
from fairseq import utils
import time
from fairseq.data import Dictionary

class KNN_Dstore(object):
    def __init__(self, args):
        self.half = args.fp16
        if hasattr(args, "decoder_embed_dim"):
            self.dimension = args.decoder_embed_dim
        else:
            self.dimension = args.knn_embed_dim
        self.k = args.k
        self.dstore_size = args.dstore_size
        self.metric_type = args.faiss_metric_type
        self.sim_func = args.knn_sim_func
        self.dstore_fp16 = args.dstore_fp16
        self.use_faiss_only = args.use_faiss_only
        self.index = self.setup_faiss(args)


    def setup_faiss(self, args):
        if not args.indexfile:
            raise ValueError('Cannot use knnlm without an index.')

        start = time.time()
        index = faiss.read_index(args.indexfile, faiss.IO_FLAG_ONDISK_SAME_DIR)
        print('Reading datastore took {} s'.format(time.time() - start))
        if args.knn_q2gpu:
            print("Moving quantizer to GPU")
            index_ivf = faiss.extract_index_ivf(index)
            quantizer = index_ivf.quantizer
            quantizer_gpu = faiss.index_cpu_to_all_gpus(quantizer, ngpu=1)
            index_ivf.quantizer = quantizer_gpu

        index.nprobe = args.probe

        if self.use_faiss_only:
            return index

        if args.dstore_fp16:
            print('Keys are fp16 and vals are int16')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float16, mode='r', shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int16, mode='r', shape=(self.dstore_size, 1))
        else:
            print('Keys are fp32 and vals are int64')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float32, mode='r', shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int, mode='r', shape=(self.dstore_size, 1))

        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if args.move_dstore_to_mem:
            print('Loading to memory...')
            start = time.time()

            if not args.no_load_keys:
                del self.keys
                self.keys_from_memmap = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float32, mode='r', shape=(self.dstore_size, self.dimension))
                self.keys = np.zeros((self.dstore_size, self.dimension), dtype=np.float16 if args.dstore_fp16 else np.float32)
                self.keys = self.keys_from_memmap[:]
                self.keys = self.keys.astype(np.float16 if args.dstore_fp16 else np.float32)

            del self.vals
            self.vals_from_memmap = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int, mode='r', shape=(self.dstore_size, 1))
            self.vals = np.zeros((self.dstore_size, 1), dtype=np.int16 if args.dstore_fp16 else np.int)
            self.vals = self.vals_from_memmap[:]
            self.vals = self.vals.astype(np.int16 if args.dstore_fp16 else np.int)
            print('Loading to memory took {} s'.format(time.time() - start))

        return index


    def get_knns(self, queries):
        start = time.time()
        dists, knns = self.index.search(queries.detach().cpu().float().numpy(), self.k)
        return dists, knns

    def dist_func(self, d, k, q, function=None):
        if not function:
            # Default behavior for L2 metric is to recompute distances.
            # Default behavior for IP metric is to return faiss distances.
            qsize = q.shape
            if self.metric_type == 'l2':
                start = time.time()
                knns_vecs = torch.from_numpy(self.keys[k]).cuda().view(qsize[0], self.k, -1)
                if self.half:
                    knns_vecs = knns_vecs.half()
                query_vecs = q.view(qsize[0], 1, qsize[1]).repeat(1, self.k, 1)
                l2 = torch.sum((query_vecs - knns_vecs.detach())**2, dim=2)
                return -1 * l2
            return d

        if function == 'dot':
            qsize = q.shape
            return (torch.from_numpy(self.keys[k]).cuda() * q.view(qsize[0], 1, qsize[1])).sum(dim=-1)

        if function == 'sqrt':
            return -1 * torch.sqrt(d)

        if function == 'do_not_recomp_l2':
            return -1 * d

        raise ValueError("Invalid knn similarity function!")

    def get_knn_log_prob(self, queries, tgt, pad_idx):
        # queries  are TxBxC
        # reshape: (TxB)xC
        qshape = queries.shape
        queries = queries.view(-1, qshape[-1])
        tgt = tgt.contiguous().view(-1)
        dists, knns = self.get_knns(queries[tgt != pad_idx])
        # (T_reducedxB)xK
        dists = torch.from_numpy(dists).cuda()
        start = time.time()
        dists = self.dist_func(dists, knns, queries[tgt != pad_idx, :], function=self.sim_func)
        probs = utils.log_softmax(dists, dim=-1)

        index_mask = torch.eq(torch.from_numpy(self.vals[knns]).long().cuda().squeeze(-1), tgt[tgt != pad_idx].unsqueeze(-1)).float()
        index_mask[index_mask == 0] = -10000 # for stability
        index_mask[index_mask == 1] = 0

        # (T_reducedxB)
        yhat_knn_prob = torch.logsumexp(probs + index_mask, dim=-1).clone()
        full_yhat_knn_prob = torch.full([qshape[0]*qshape[1]], -10000).cuda()
        full_yhat_knn_prob[tgt != pad_idx] = yhat_knn_prob

        # TxBx1
        return full_yhat_knn_prob.view(qshape[0], qshape[1], 1)

    def get_knn_scores_per_step(self, queries, vocab_size, pad_idx, use_dtype=torch.float32, save_knns=False, knn_temp=1.0):
        qshape = queries.shape
        queries = queries.view(-1, qshape[-1])
        start_time = time.time()
        dists, knns = self.get_knns(queries)
        #print(f"Querying knns: {time.time()-start_time}s")
        # (Bxbeam_size)xK
        dists = torch.from_numpy(dists).type(dtype=use_dtype).cuda()
        #print(dists)
        # TODO(urvashik): update to use full precision keys
        #dists = -1 * dists
        dists = self.dist_func(dists, knns, queries, function=self.sim_func)
        #print(dists)
        dists.div_(knn_temp)
        probs = utils.log_softmax(dists, dim=-1).type(dtype=use_dtype)
        #print(probs)

        # (Bxbeam_size)xK
        if self.use_faiss_only:
            indices = torch.from_numpy(knns).long().cuda()
        else:
            indices = torch.from_numpy(self.vals[knns]).long().cuda()
        indices = indices.view(queries.shape[0], self.k)
        #print(indices)

        ## TRYING SOMETHING OUT
        unique_indices, mapping = torch.unique(indices, return_inverse=True)
        # (Bxbeam)xKxn where n = num unique vals in indices
        knn_scores_by_index = torch.ones([indices.shape[0], indices.shape[1], len(unique_indices)], dtype=use_dtype).cuda()
        knn_scores_by_index[:] = -10000 #-math.inf
        knn_vals_by_index = torch.ones([indices.shape[0], indices.shape[1], len(unique_indices)]).long().cuda()
        knn_vals_by_index[:] = pad_idx

        # (Bxbeam)x1xK
        indices = indices.unsqueeze(2)
        probs = probs.unsqueeze(2)
        mapping = mapping.unsqueeze(2)
        knn_scores_by_index.scatter_(dim=2, index=mapping, src=probs)
        knn_vals_by_index.scatter_(dim=2, index=mapping, src=indices)
        # (Bxbeam)xn
        knn_scores_by_index = knn_scores_by_index.logsumexp(dim=1)
        knn_vals_by_index = knn_vals_by_index.max(dim=1)[0]
        full_knn_scores = torch.ones([queries.shape[0], vocab_size], dtype=use_dtype).cuda()
        full_knn_scores[:] = -10000 #-math.inf
        full_knn_scores.scatter_(dim=1, index=knn_vals_by_index, src=knn_scores_by_index)
        ## TRYING SOMETHING OUT

        ## FOR DEBUGGING
#        full_knn_scores = torch.ones([queries.shape[0], vocab_size], dtype=torch.float32).cuda()
#        full_knn_scores[:] = -math.inf
#        for i in range(probs.shape[0]):
#            retr_vals = torch.unique(indices[i])
#            for j in retr_vals:
#                locs = (indices[i] == j).nonzero()
#                full_knn_scores[i, j] = torch.logsumexp(probs[i, locs].view(-1), dim=0)
        ## FOR DEBUGGING
        if save_knns:
            return full_knn_scores, [torch.from_numpy(knns).long().cuda(), probs.squeeze(-1), indices.squeeze(-1)]

        return full_knn_scores


