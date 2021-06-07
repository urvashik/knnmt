import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dstore_mmap', type=str, help='prefix to read/write the keys/values')
parser.add_argument('--num_datasets', type=int, help='number of datasets')
parser.add_argument('--dataset_start', type=int, default=0, help='index less than num_datasets to start at')
parser.add_argument('--num_langs', type=int, default=1, help="number of languages")
parser.add_argument('--langs', type=str, action='append', help="languages")
parser.add_argument('--size', type=int, help='size of each individual memmap (should be identical)')
parser.add_argument('--dimension', type=int, default=1024)
args = parser.parse_args()

print(args)

dstore_keys = np.memmap(args.dstore_mmap + ".subset_keys.npy", dtype=np.float32, mode='w+', shape=(args.num_datasets*args.size*args.num_langs, args.dimension))
dstore_vals = np.memmap(args.dstore_mmap + ".subset_vals.npy", dtype=np.int, mode='w+', shape=(args.num_datasets*args.size*args.num_langs, 1))

if args.langs:
    for lang_idx, lang in enumerate(args.langs):
        # num_datasets in this case is number of files for each language
        offset = args.num_datasets * lang_idx * args.size # eg: 2 * 1 * 500k
        for i in range(args.num_datasets):
            filename = args.dstore_mmap + ".subset." + lang + "." + str(i)
            print(filename)
            keys = np.memmap(filename + "_keys.npy", dtype=np.float32, mode='r', shape=(args.size, args.dimension))
            vals = np.memmap(filename + "_vals.npy", dtype=np.int, mode='r', shape=(args.size, 1))
            dstore_keys[offset+(i*args.size):offset+(i*args.size+args.size)] = keys[:]
            dstore_vals[offset+(i*args.size):offset+(i*args.size+args.size)] = vals[:]

    print('merged %d keys/values' % (args.size*args.num_datasets*args.num_langs))

else:
    for i in range(args.num_datasets):
        filename = args.dstore_mmap + ".subset." + str(i)
        keys = np.memmap(filename + "_keys.npy", dtype=np.float32, mode='r', shape=(args.size, args.dimension))
        vals = np.memmap(filename + "_vals.npy", dtype=np.int, mode='r', shape=(args.size, 1))
        dstore_keys[i*args.size:i*args.size+args.size] = keys[:]
        dstore_vals[i*args.size:i*args.size+args.size] = vals[:]

    print('merged %d keys/values' % (args.size*args.num_datasets))
