import argparse
import os
import numpy as np
import faiss
import time


parser = argparse.ArgumentParser()
parser.add_argument('--dstore_mmap', type=str, help='memmap where keys and vals are stored')
parser.add_argument('--dstore_size', type=int, help='number of items saved in the datastore memmap')
parser.add_argument('--dimension', type=int, default=1024, help='Size of each key')
parser.add_argument('--dstore_fp16', default=False, action='store_true')
parser.add_argument('--seed', type=int, default=1, help='random seed for sampling the subset of vectors to train the cache')
parser.add_argument('--ncentroids', type=int, default=4096, help='number of centroids faiss should learn')
parser.add_argument('--code_size', type=int, default=64, help='size of quantized vectors')
parser.add_argument('--probe', type=int, default=8, help='number of clusters to query')
parser.add_argument('--train_index', type=str, help='file to write the faiss index')
parser.add_argument('--from_subset', action='store_true', default=False, help='training the index from a mmap that only holds keys needed for training.')
parser.add_argument('--gpu', action='store_true', default=False, help='training the index from a mmap that only holds keys needed for training.')
args = parser.parse_args()

print(args)

if args.dstore_fp16:
    keys = np.memmap(args.dstore_mmap+'_keys.npy', dtype=np.float16, mode='r', shape=(args.dstore_size, args.dimension))
    vals = np.memmap(args.dstore_mmap+'_vals.npy', dtype=np.int16, mode='r', shape=(args.dstore_size, 1))
else:
    keys = np.memmap(args.dstore_mmap+'_keys.npy', dtype=np.float32, mode='r', shape=(args.dstore_size, args.dimension))
    vals = np.memmap(args.dstore_mmap+'_vals.npy', dtype=np.int, mode='r', shape=(args.dstore_size, 1))

# Initialize faiss index

quantizer = faiss.IndexFlatL2(args.dimension)
index = faiss.IndexIVFPQ(quantizer, args.dimension,
    args.ncentroids, args.code_size, 8)
index.nprobe = args.probe

print('Training Index')
if args.from_subset:
    if args.gpu:
        print("Moving index to gpu before training")
        clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(args.dimension), ngpu=1)
        index.clustering_index = clustering_index
    start = time.time()
    index.train(keys[:].astype(np.float32))
    print('Training took {} s'.format(time.time() - start))
else:
    np.random.seed(args.seed)
    random_sample = np.random.choice(np.arange(vals.shape[0]), size=[min(1000000, vals.shape[0])], replace=False)
    random_sample = np.sort(random_sample)
    start = time.time()
    # Faiss does not handle adding keys in fp16 as of writing this.
    index.train(keys[random_sample].astype(np.float32))
    print('Training took {} s'.format(time.time() - start))

print('Writing index after training')
start = time.time()
faiss.write_index(index, args.train_index)
print('Writing index took {} s'.format(time.time()-start))
