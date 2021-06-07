import submitit
import argparse
import itertools
import os
import math
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--log-folder', type=str, help='Folder to log stdout and stderr')
parser.add_argument('--slurm-name', type=str)
parser.add_argument('--slurm-mem-gb', type=int)
parser.add_argument('--slurm-timeout-min', type=int)
parser.add_argument('--slurm-partition', type=str)
parser.add_argument('--slurm-constraint', type=str)
parser.add_argument('--slurm-nodes', type=int)
parser.add_argument('--slurm-ntasks-per-node', type=int)
parser.add_argument('--slurm-cpus-per-task', type=int)
parser.add_argument('--slurm-gpus-per-node', type=int)
parser.add_argument('--array-parallelism', type=int)
parser.add_argument('--less-mem-gb', type=int)
# Model params
parser.add_argument('--bytes-per-token', type=int, default=2, help='check the binary file size to be calculate (size of file / number of tokens), 2 when using the wmt19 vocab')
parser.add_argument('--model', type=str, help='pytorch checkpoint to use')
parser.add_argument('--bpe', type=str, help='fastbpe or subword_nmt')
parser.add_argument('--bpecodes', type=str, help='path to codes')
# dstore saving params
parser.add_argument('--dstore-size', type=int, action='append', default=None, help='list of sizes of dstores in number of tokens')
parser.add_argument('--save-data', type=str, action='append', help='list of datasets to save in dstore')
parser.add_argument('--binfile', type=str, action='append', help='list of bin filenames to get file size')
parser.add_argument('--num-shards', type=int, action='append', help='number of shards for each file')
parser.add_argument('--dstore-mmap', type=str, help='dstore location for datasets')
parser.add_argument('--num-for-training', type=int, help='number of items to save per dataset for training the index')
parser.add_argument('--code-size', type=int, default=64, help='size that vectors are quantized to')
parser.add_argument('--ncentroids', type=int, action='append', help='number of faiss clusters')
parser.add_argument('--train-index', type=str, action='append',  help='list of files to use for the trained faiss index, must be the same length as ncentroids')
parser.add_argument('--faiss-index', type=str, action='append', help='list of files to use for the faiss index')
parser.add_argument('--write-merged-index', type=str, action='append', help='list of files to use for the faiss index')
parser.add_argument('--corpus-identifiers', type=str, action='append', help='list of ids to use for the distributed faiss indices')
# run job params
parser.add_argument('--save-job', action='store_true', default=False)
parser.add_argument('--merge-dstore-job', action='store_true', default=False)
parser.add_argument('--train-index-job', action='store_true', default=False)
parser.add_argument('--add-keys-job', action='store_true', default=False)
parser.add_argument('--merge-index-job', action='store_true', default=False)
args = parser.parse_args()

executor = submitit.AutoExecutor(folder=args.log_folder)
executor.update_parameters(name=args.slurm_name,
        mem_gb=args.slurm_mem_gb,
        timeout_min=args.slurm_timeout_min,
        slurm_partition=args.slurm_partition,
        slurm_constraint=args.slurm_constraint,
        nodes=args.slurm_nodes,
        tasks_per_node=args.slurm_ntasks_per_node,
        cpus_per_task=args.slurm_cpus_per_task,
        gpus_per_node=args.slurm_gpus_per_node,
        slurm_signal_delay_s=120,
        slurm_array_parallelism=args.array_parallelism,
    )

dstore_size = args.dstore_size
if not dstore_size:
    dstore_size = []
    assert len(args.save_data) == len(args.binfile)
    for dataset, binfile in zip(args.save_data, args.binfile):
        filestats = os.stat(os.path.join(dataset, binfile))
        size = filestats.st_size / args.bytes_per_token # 2 bytes per token for most
        dstore_size.append(int(size))

        print("%s with num tokens %d" % (dataset, size))

if args.slurm_partition == 'priority':
    executor.update_parameters(comment='end of internship 9/11')

# SAVE KEYS/VALUES SUBSET FOR TRAINING
if args.save_job:
    save_jobs = []
    with executor.batch():
        for dataset_idx, curr_save_data in enumerate(args.save_data):
            curr_save_subset_mmap = args.dstore_mmap + ".subset." + str(dataset_idx)
            save_cmd = f"python fairseq_cli/generate.py {curr_save_data} --gen-subset train --path {args.model} --beam 5 --remove-bpe --bpe {args.bpe} --bpe-codes {args.bpecodes} --tokenizer moses --moses-source-lang de --moses-target-lang en --sacrebleu --score-reference --dstore-mmap {curr_save_subset_mmap} --knn-keytype last_ffn_input  --model-overrides {{\'knn_keytype\':\'last_ffn_input\'}} --save-knn-dstore --save-knn-subset --save-knn-subset-num {args.num_for_training} --quiet"
            print(save_cmd)
            function = submitit.helpers.CommandFunction(save_cmd.split())
            job = executor.submit(function)
            save_jobs.append(job)

    for job in save_jobs:
        job.result()
# SAVE KEYS/VALUES SUBSET FOR TRAINING

# MERGE SUBSET KEYS/VALUES
if args.merge_dstore_job:
    print("Merging subsets saved for training")
    num_datasets = len(args.save_data)
    if args.slurm_partition != 'priority':
        executor.update_parameters(slurm_partition='dev', gpus_per_node=0)
    else:
        executor.update_parameters(gpus_per_node=0)
    merge_subset_cmd = f"python merge_subset_dstores.py --dstore_mmap {args.dstore_mmap} --num_datasets {num_datasets} --size {args.num_for_training}"
    print(merge_subset_cmd)
    function = submitit.helpers.CommandFunction(merge_subset_cmd.split())
    job = executor.submit(function)
    job.result()
# MERGE SUBSET KEYS/VALUES

# TRAIN INDEX
if args.train_index_job:
    train_jobs = []
    assert len(args.ncentroids) == len(args.train_index)
    dstore_mmap = args.dstore_mmap + ".subset"
    size = len(args.save_data) * args.num_for_training
    if args.slurm_partition != 'priority':
        executor.update_parameters(slurm_partition='dev', gpus_per_node=1)
    else:
        executor.update_parameters(gpus_per_node=1)
    with executor.batch():
        for ncentroid, train_index in zip(args.ncentroids, args.train_index):
            print("Training index with %d centroids" % (ncentroid))
            train_cmd = f"python train_index.py --dstore_mmap {dstore_mmap} --dstore_size {size} --dimension 1024 --code_size {args.code_size} --ncentroids {ncentroid} --train_index {train_index} --from_subset --gpu"
            print(train_cmd)
            function = submitit.helpers.CommandFunction(train_cmd.split())
            job = executor.submit(function)
            train_jobs.append(job)

    for job in train_jobs:
        job.result()
# TRAIN INDEX

# Add keys to an already trained index.
# This is done from multiple files, passed through the command line using append.
if args.add_keys_job:
    print("Adding keys to the faiss index")
    executor.update_parameters(slurm_partition=args.slurm_partition, gpus_per_node=args.slurm_gpus_per_node)
    add_jobs = []
    assert len(args.train_index) == len(args.faiss_index)
    assert len(args.save_data) == len(args.corpus_identifiers)
    assert len(args.save_data) == len(args.num_shards)
    with executor.batch():
        total_added = 0
        for dataset_idx, (curr_save_data, curr_dstore_size, num_shards) in enumerate(zip(args.save_data, dstore_size, args.num_shards)):
            print("Saving %s, of size %d" % (curr_save_data, curr_dstore_size))
            # iterations it will take to add all keys to index
            index_id = 0 # which index is being written
            train_index = " ".join([f"--trained-index {tindex}" for tindex in args.train_index])
            for shard_idx in range(num_shards):
                write_index = " ".join([f"--write-index {faiss_index}.{args.corpus_identifiers[dataset_idx]}.{index_id}" for faiss_index in args.faiss_index])
                add_cmd = f"python fairseq_cli/generate.py {curr_save_data} --gen-subset train --path {args.model} --beam 5 --remove-bpe --bpe {args.bpe} --bpe-codes {args.bpecodes} --tokenizer moses --moses-source-lang de --moses-target-lang en --sacrebleu --score-reference --knn-keytype last_ffn_input  --model-overrides {{\'knn_keytype\':\'last_ffn_input\'}} --save-knn-dstore --knn-add-to-idx --num-shards {num_shards} --shard-id {shard_idx} {train_index} {write_index} --quiet --knn-q2gpu"
                print(add_cmd)
                function = submitit.helpers.CommandFunction(add_cmd.split())
                job = executor.submit(function)
                add_jobs.append(job)
                index_id += 1 # remember this is 1 greater than the actual ids for indices, i.e. there are index_id number of indices but the last one is index_id - 1.

            print("Number of indices for this dataset %d" % (index_id))
        print("Total keys meant to be added = %d" % (sum(dstore_size)))


    for job in add_jobs:
        job.result()
# Add keys to an already trained index.

# MERGE FAISS INDICES
if args.merge_index_job:
    merge_index_jobs = []
    if args.slurm_partition != 'priority':
        executor.update_parameters(slurm_partition='dev', gpus_per_node=0)
    else:
        executor.update_parameters(gpus_per_node=0)
    executor.update_parameters(slurm_constraint='volta32gb', cpus_per_task=10)
    print('Merging indices')
    corpus_identifiers = " ".join([f"--corpus_identifiers {cid}" for cid in args.corpus_identifiers])
    num_shards = " ".join([f"--num_shards_per_file {ns}" for ns in args.num_shards])
    with executor.batch():
        for tindex, findex, wmindex in zip(args.train_index, args.faiss_index, args.write_merged_index):
            merge_idx_cmd = f"python merge_index.py --faiss_index {findex} --train_index {tindex} {corpus_identifiers} --write_index {wmindex} {num_shards}"
            print(merge_idx_cmd)
            function = submitit.helpers.CommandFunction(merge_idx_cmd.split())
            job = executor.submit(function)
            merge_index_jobs.append(job)

    for job in merge_index_jobs:
        job.result()
# MERGE FAISS INDICES
