# Nearest Neighbor Machine Translation

This repository is a fork of the [Fairseq](https://github.com/pytorch/fairseq) repository from July 2020, and used PyTorch 1.4.0 and Python 3.6.  

This code pertains to the ICLR 2021 paper: [Nearest Neighbor Machine Translation](https://arxiv.org/pdf/2010.00710.pdf). If you use this code or results from our paper, please cite:  

```
@inproceedings{khandelwal2021nearest,
  title={Nearest Neighbor Machine Translation},
  author={Khandelwal, Urvashi and Fan, Angela and Jurafsky, Dan and Zettlemoyer, Luke and Lewis, Mike},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}
```

__Note__: Please note that this is an early release before cleaning for readability and contains a lot of extra code that did not contribute to the experiments of the paper. I have just graduated and will be offline for a while, and will provide a cleaner version of the code in the future. Also note, this code currently relies on the use of Slurm and a tool called Submitit. If your setup does not use Slurm, you may need to modify some of the scripts before using them.

## Sample commands

The following sample commands require that you install fairseq (`pip install --editable .` after pulling this project directory), [faiss](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md), [moses](https://github.com/moses-smt/mosesdecoder) and [fastBPE](https://github.com/glample/fastBPE).  

Download the [multi-domains data](https://github.com/roeeaharoni/unsupervised-domain-clusters) into the `examples/translation/` folder. The commands below use the medical data.

Download the [WMT19 German-English](https://github.com/pytorch/fairseq/tree/8b9eaacf6b2d502cd7886dd7bf702a46ab37f058/examples/translation) (wmt19.de-en)  model from fairseq, along with the dictionaries and the bpecodes.  

Preprocess the data using the `prepare-domadapt.sh` script in the `examples/translation` directory, but be sure to set the paths for the moses scripts, fastBPE, the model dictionaries and the bpecodes first:
```
bash prepare-domadapt.sh medical
```
If you have to rerun the script, make sure you delete the `tok` and `bpe` files first because the script will append to the existing files.

Then prepare the binary files from the knnmt root directory. Be sure to create the `data-bin/medical.de-en` directory and copy the German dictionary there:  
```
python fairseq_cli/preprocess.py --source-lang de --target-lang en --trainpref examples/translation/medical/train.bpe.filtered --validpref examples/translation/medical/dev.bpe --testpref examples/translation/medical/test.bpe --destdir data-bin/medical.de-en/ --srcdict data-bin/medical.de-en/dict.de.txt --joined-dictionary
```

Creating the datastore is governed by `master_script.py` which contains a lot of logic for parallelizing things which is not useful/reusable. If you do not want to use the script, you can figure out the steps from it and execute them separately -- training the index and adding all the key-value pairs to it.  

The master script uses slurm and a tool called submitit to execute jobs and requires a whole host of slurm related parameters to be provided. If your setup uses slurm, please determine the appropriate settings for those parameters. If it does not use slurm, you would need to modify the script to make it compatible with your setup.

__Note__: In this work, we create the datastore without saving the full-precision keys or the values to a separate memory map, which is different from how we did this in [kNN-LM](https://github.com/urvashik/knnlm). The key-value pairs are added directly to the faiss index.

Create the `dstores/medical` directory before executing the following commands.  

```
python master_script.py --log-folder logs/medical/build_index --slurm-name med --bytes-per-token 2 --model checkpoints/wmt19.de-en/model.pt --bpe fastbpe --bpecodes checkpoints/wmt19.de-en/bpecodes --save-data data-bin/medical.de-en/ --binfile train.de-en.en.bin --num-shards 1 --dstore-mmap dstores/medical/index_only --num-for-training 1000000 --code-size 64 --ncentroids 4096 --train-index dstores/medical/index_only.4096.index.trained --save-job --merge-dstore-job --train-index-job  

python master_script.py --log-folder logs/medical/build_index/add_merge_index --slurm-name med --bytes-per-token 2 --model checkpoints/wmt19.de-en/model.pt --bpe fastbpe --bpecodes checkpoints/wmt19.de-en/bpecodes --save-data data-bin/medical.de-en/ --binfile train.de-en.en.bin --num-shards 1 --dstore-mmap dstores/medical/index_only --num-for-training 1000000 --code-size 64 --ncentroids 4096 --train-index dstores/medical/index_only.4096.index.trained  --faiss-index dstores/medical/index_only.4096.index --write-merged-index dstores/medical/index_only.4096.index --corpus-identifiers med --add-keys-job --merge-index-job
```

Then, to evaluate the model with and without knnmt:
```
python fairseq_cli/generate.py data-bin/medical.de-en/ --gen-subset valid --path checkpoints/wmt19.de-en/model.pt --beam 5 --remove-bpe --bpe fastbpe --bpe-codes checkpoints/wmt19.de-en/bpecodes --tokenizer moses --moses-source-lang de --moses-target-lang en --quiet --sacrebleu  

python fairseq_cli/generate.py data-bin/medical.de-en/ --gen-subset valid --path checkpoints/wmt19.de-en/model.pt --beam 5 --remove-bpe --bpe fastbpe --bpe-codes checkpoints/wmt19.de-en/bpecodes --tokenizer moses --moses-source-lang de --moses-target-lang en --quiet --sacrebleu --knnmt --k 64 --probe 32 --indexfile dstores/medical/index_only.4096.index --model-overrides "{'knn_keytype': 'last_ffn_input'}" --knn-keytype last_ffn_input --knn-embed-dim 1024 --no-load-keys  --knn-temp 10 --knn-sim-func do_not_recomp_l2 --lmbda 0.8 --use-faiss-only
```
