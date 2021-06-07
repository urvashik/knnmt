import argparse
import os
import numpy as np
import faiss
import time


parser = argparse.ArgumentParser()
parser.add_argument('--faiss_index', type=str, help='file to write the faiss index')
parser.add_argument('--write_index', type=str, help='file to write the faiss index')
parser.add_argument('--train_index', type=str, help='corresponding trained index')
parser.add_argument('--corpus_identifiers', type=str, action='append', help='faiss index corpus identifier')
parser.add_argument('--num_shards_per_file', type=int, action='append', help='number of shards')
parser.add_argument('--num_files_per_cid', type=int, default=1, help='for mmt more than 1 file')
parser.add_argument('--start_num_files_per_cid', type=int, default=0, help='if num splits starts at a value other than 0')
args = parser.parse_args()

print(args)

ivfs = []
for cid, num_shards_per_file in zip(args.corpus_identifiers, args.num_shards_per_file):
    print('Reading for dataset %s' % cid)
    for fileidx in range(args.start_num_files_per_cid, args.num_files_per_cid):
        for k in range(num_shards_per_file):
            if args.num_files_per_cid > 1:
                filename = args.faiss_index + "." + cid + "." + str(fileidx) + "." + str(k)
            else:
                filename = args.faiss_index + "." + cid + "." + str(k)
            if not os.path.exists(filename):
                raise ValueError("File does not exist", filename)
            print("Reading index %s" % filename)
            index = faiss.read_index(filename, faiss.IO_FLAG_MMAP)
            ivfs.append(index.invlists)
            # avoid that the invlists get deallocated with the index
            index.own_invlists = False

index = faiss.read_index(args.train_index)
# prepare the output inverted lists. They will be written
# to merged_index.ivfdata
invlists = faiss.OnDiskInvertedLists(
        index.nlist, index.code_size,
        args.write_index + ".merged_index.ivfdata")

# merge all the inverted lists
ivf_vector = faiss.InvertedListsPtrVector()
for ivf in ivfs:
    ivf_vector.push_back(ivf)

print("merge %d inverted lists" % ivf_vector.size())
ntotal = invlists.merge_from(ivf_vector.data(), ivf_vector.size())

# now replace the inverted lists in the output index
index.ntotal = ntotal
index.replace_invlists(invlists)

faiss.write_index(index, args.write_index)
print("Size of datastore = %d" % index.ntotal)
