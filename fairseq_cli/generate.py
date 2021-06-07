#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import logging
import math
import os
import sys
import time

import numpy as np
import pickle

import torch
import faiss

from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.data import encoders


def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.dataset_impl == 'raw', \
        '--replace-unk requires a raw text dataset (--dataset-impl=raw)'

    if args.results_path is not None:
        os.makedirs(args.results_path, exist_ok=True)
        output_path = os.path.join(args.results_path, 'generate-{}.txt'.format(args.gen_subset))
        with open(output_path, 'w', buffering=1, encoding='utf-8') as h:
            return _main(args, h)
    else:
        return _main(args, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, 'symbols_to_strip_from_output'):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def _main(args, output_file):
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        stream=output_file,
    )
    logger = logging.getLogger('fairseq_cli.generate')

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    logger.info(args)

    # Fix seed for stochastic decoding
    if args.seed is not None and not args.no_seed_provided:
        np.random.seed(args.seed)
        utils.set_torch_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    logger.info('loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args.path),
        arg_overrides=eval(args.model_overrides),
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
    )

    # Optimize ensemble for generation
    for model in models:
        model.prepare_for_inference_(args)
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
         task.max_positions(),
        *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        default_log_format=('tqdm' if not args.no_progress_bar else 'none'),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(models, args)

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    ## knn saving code
    if args.save_knn_dstore:
        print('keytype being saved:', args.knn_keytype)
        if args.knn_start > -1:
            chunk_size = 100000
            if args.dstore_fp16:
                print('Saving fp16')
                dstore_keys = np.zeros([chunk_size, model.decoder.embed_dim], dtype=np.float16)
                dstore_vals = np.zeros([chunk_size, 1], dtype=np.int16)
            else:
                print('Saving fp32')
                dstore_keys = np.zeros([chunk_size, model.decoder.embed_dim], dtype=np.float32)
                dstore_vals = np.zeros([chunk_size, 1], dtype=np.int)

        else:
            assert not (args.save_knn_subset and args.knn_add_to_idx)
            dstore_size = args.dstore_size
            if args.save_knn_subset:
                dstore_size = args.save_knn_subset_num
            if args.dstore_fp16:
                print('Saving fp16')
                if args.knn_add_to_idx:
                    faiss_indices = []
                    for tindex in args.trained_index:
                        print("Reading trained index from %s" % tindex)
                        faiss_indices.append(faiss.read_index(tindex))
                        if args.knn_q2gpu:
                            assert len(args.trained_index) == 1
                            print("Moving quantizer to GPU")
                            index_ivf = faiss.extract_index_ivf(faiss_indices[0])
                            quantizer = index_ivf.quantizer
                            quantizer_gpu = faiss.index_cpu_to_all_gpus(quantizer, ngpu=1)
                            index_ivf.quantizer = quantizer_gpu
                else:
                    dstore_keys = np.memmap(args.dstore_mmap+'_keys.npy', dtype=np.float16, mode='w+', shape=(dstore_size, model.decoder.embed_dim))
                    dstore_vals = np.memmap(args.dstore_mmap+'_vals.npy', dtype=np.int16, mode='w+', shape=(dstore_size, 1))
            else:
                print('Saving fp32')
                if args.knn_add_to_idx:
                    faiss_indices = []
                    for tindex in args.trained_index:
                        print("Reading trained index from %s" % tindex)
                        faiss_indices.append(faiss.read_index(tindex))
                        if args.knn_q2gpu:
                            assert len(args.trained_index) == 1
                            print("Moving quantizer to GPU")
                            index_ivf = faiss.extract_index_ivf(faiss_indices[0])
                            quantizer = index_ivf.quantizer
                            quantizer_gpu = faiss.index_cpu_to_all_gpus(quantizer, ngpu=1)
                            index_ivf.quantizer = quantizer_gpu
                else:
                    dstore_keys = np.memmap(args.dstore_mmap+'_keys.npy', dtype=np.float32, mode='w+', shape=(dstore_size, model.decoder.embed_dim))
                    dstore_vals = np.memmap(args.dstore_mmap+'_vals.npy', dtype=np.int, mode='w+', shape=(dstore_size, 1))

        dstore_idx = 0
        total_saved = 0
        knn_num_samples_proc = 0
        to_skip = -1
        if args.knn_start > -1:
            to_skip = args.knn_start # examples
            start_pos = 0
        if args.knn_add_to_idx:
            adding_to_faiss = 0
        # save the sample ids and the lengths for backtracking the neighbors
        sample_order_lens = [[],[]]
    if args.knnmt and args.save_knns:
        to_save_objects = []
    ## knn saving code

    # Generate and compute BLEU score
    scorer = scoring.scoring_utils.build_scorer(args, tgt_dict)

    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    for idx, sample in enumerate(progress):
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if 'net_input' not in sample:
            continue

        ## For processing in parallel
        if args.save_knn_dstore and to_skip > 0:
            num_samples = sample['target'].shape[0]
            if to_skip - num_samples > 0:
                to_skip -= num_samples
                target_tokens = utils.strip_pad(sample['target'], tgt_dict.pad()).int().cpu()
                start_pos += len(target_tokens)
                continue

            for i, sample_id in enumerate(sample['id'].tolist()):
                if to_skip > 0:
                    to_skip -= 1
                    target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()
                    start_pos += len(target_tokens)
                else:
                    tgt_tokens = utils.strip_pad(sample['target'][i:], tgt_dict.pad()).int().cpu()
                    new_sample = {
                            'id': sample['id'][i:],
                            'nsentences': len(sample['id'][i:]),
                            'ntokens': len(tgt_tokens),
                            'net_input': {
                                'src_tokens': sample['net_input']['src_tokens'][i:],
                                'src_lengths': sample['net_input']['src_lengths'][i:],
                                'prev_output_tokens': sample['net_input']['prev_output_tokens'][i:],
                            },
                            'target': sample['target'][i:]

                    }
                    sample = new_sample
                    break

            print('Starting the saving at location %d in the mmap' % start_pos)
        ## For processing in parallel

        prefix_tokens = None
        if args.prefix_size > 0:
            prefix_tokens = sample['target'][:, :args.prefix_size]

        #print('target', sample['target'].shape)
        gen_timer.start()
        hypos = task.inference_step(generator, models, sample, prefix_tokens)
        #exit()
        num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
        #print(hypos[0][0]['tokens'])
        #exit(0)
        gen_timer.stop(num_generated_tokens)

        if args.knn_add_to_idx:
            saving = sample['ntokens']
            if args.drop_lang_tok:
                saving = sample['ntokens'] - sample['target'].shape[0]
            keys = np.zeros([saving, model.decoder.embed_dim], dtype=np.float32)
            addids = np.zeros([saving], dtype=np.int)
            save_idx = 0

        for i, sample_id in enumerate(sample['id'].tolist()):
            loop_start = time.time()
            has_target = sample['target'] is not None
            #print(sample['target'][i])

            # Remove padding
            if 'src_tokens' in sample['net_input']:
                src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
            else:
                src_tokens = None

            target_tokens = None
            if has_target:
                target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

            #print(len(hypos))
            #print(hypos[i][0]['tokens'].shape)
            #print(len(target_tokens))
            #print(hypos[i][0]['tokens'])
            ## knn saving code
            if args.save_knn_dstore:
                hypo = hypos[i][0]
                num_items = len(hypo['tokens'])
                #print(num_items, hypo['dstore_keys_mt'].shape)
                #print(hypo['tokens'])
                #print(hypo['dstore_keys_mt'])
                #exit(0)
                #sample_order_lens[0].append(sample_id)
                #sample_order_lens[1].append(num_items)
                #if dstore_idx + shape[0] > args.dstore_size:
                #    shape = [args.dstore_size - dstore_idx]
                #    hypo['dstore_keys_mt'] = hypo['dstore_keys_mt'][:shape[0]]
                if args.knn_start > -1:
                    if dstore_idx + num_items > dstore_keys.shape[0]:
                        if args.dstore_fp16:
                            dstore_keys = np.concatenate([dstore_keys, np.zeros([chunk_size, model.decoder.embed_dim], dtype=np.float16)], axis=0)
                            dstore_vals = np.concatenate([dstore_vals, np.zeros([chunk_size, 1], dtype=np.int16)], axis=0)
                        else:
                            dstore_keys = np.concatenate([dstore_keys, np.zeros([chunk_size, model.decoder.embed_dim], dtype=np.float32)], axis=0)
                            dstore_vals = np.concatenate([dstore_vals, np.zeros([chunk_size, 1], dtype=np.int)], axis=0)

                skip = 0
                if args.drop_lang_tok:
                    skip += 1

                if args.save_knn_subset:
                    if total_saved + num_items - skip > args.save_knn_subset_num:
                        num_items = args.save_knn_subset_num - total_saved + skip

                if args.knn_add_to_idx:
                    keys[save_idx:save_idx+num_items-skip] = hypo['dstore_keys_mt'][skip:num_items].view(
                            -1, model.decoder.embed_dim).cpu().numpy().astype(np.float32)
                    addids[save_idx:save_idx+num_items-skip] = hypo['tokens'][skip:num_items].view(
                            -1).cpu().numpy().astype(np.int)
                    save_idx += num_items - skip

                if not args.knn_add_to_idx:
                    if args.dstore_fp16:
                        dstore_keys[dstore_idx:num_items-skip+dstore_idx] = hypo['dstore_keys_mt'][skip:num_items].view(
                                -1, model.decoder.embed_dim).cpu().numpy().astype(np.float16)
                        dstore_vals[dstore_idx:num_items-skip+dstore_idx] = hypo['tokens'][skip:num_items].view(
                                -1, 1).cpu().numpy().astype(np.int16)
                    else:
                        dstore_keys[dstore_idx:num_items-skip+dstore_idx] = hypo['dstore_keys_mt'][skip:num_items].view(
                                -1, model.decoder.embed_dim).cpu().numpy().astype(np.float32)
                        dstore_vals[dstore_idx:num_items-skip+dstore_idx] = hypo['tokens'][skip:num_items].view(
                                -1, 1).cpu().numpy().astype(np.int)

                dstore_idx += num_items - skip
                total_saved += num_items - skip
                knn_num_samples_proc += 1
            ## knn saving code
            if args.score_reference:
                continue

            ## error analysis knnmt: save knns, vals and probs
            if args.knnmt and args.save_knns:
                to_save_objects.append(
                        {
                            "id": sample_id,
                            "src": src_tokens,
                            "tgt": target_tokens,
                            "hypo": hypos[i],
                        }
                    )
            ## error analysis knnmt: save knns, vals and probs

            # Either retrieve the original sentences or regenerate them from tokens.
            if align_dict is not None:
                src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
            else:
                if src_dict is not None:
                    src_str = src_dict.string(src_tokens, args.remove_bpe)
                else:
                    src_str = ""
                #print(get_symbols_to_strip_from_output(generator))
                if has_target:
                    target_str = tgt_dict.string(
                        target_tokens,
                        args.remove_bpe,
                        escape_unk=True,
                        extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                    )

            src_str = decode_fn(src_str)
            if has_target:
                target_str = decode_fn(target_str)

            if not args.quiet:
                if src_dict is not None:
                    print('S-{}\t{}'.format(sample_id, src_str), file=output_file)
                if has_target:
                    print('T-{}\t{}'.format(sample_id, target_str), file=output_file)

            # Process top predictions
            for j, hypo in enumerate(hypos[i][:args.nbest]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )
                detok_hypo_str = decode_fn(hypo_str)
                if not args.quiet:
                    score = hypo['score'] / math.log(2)  # convert to base 2
                    # original hypothesis (after tokenization and BPE)
                    print('H-{}\t{}\t{}'.format(sample_id, score, hypo_str), file=output_file)
                    # detokenized hypothesis
                    print('D-{}\t{}\t{}'.format(sample_id, score, detok_hypo_str), file=output_file)
                    print('P-{}\t{}'.format(
                        sample_id,
                        ' '.join(map(
                            lambda x: '{:.4f}'.format(x),
                            # convert from base e to base 2
                            hypo['positional_scores'].div_(math.log(2)).tolist(),
                        ))
                    ), file=output_file)

                    if args.print_alignment:
                        print('A-{}\t{}'.format(
                            sample_id,
                            ' '.join(['{}-{}'.format(src_idx, tgt_idx) for src_idx, tgt_idx in alignment])
                        ), file=output_file)

                    if args.print_step:
                        print('I-{}\t{}'.format(sample_id, hypo['steps']), file=output_file)

                    if getattr(args, 'retain_iter_history', False):
                        for step, h in enumerate(hypo['history']):
                            _, h_str, _ = utils.post_process_prediction(
                                hypo_tokens=h['tokens'].int().cpu(),
                                src_str=src_str,
                                alignment=None,
                                align_dict=None,
                                tgt_dict=tgt_dict,
                                remove_bpe=None,
                            )
                            print('E-{}_{}\t{}'.format(sample_id, step, h_str), file=output_file)

                # Score only the top hypothesis
                if has_target and j == 0:
                    if align_dict is not None or args.remove_bpe is not None:
                        # Convert back to tokens for evaluation with unk replacement and/or without BPE
                        target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                        hypo_tokens = tgt_dict.encode_line(detok_hypo_str, add_if_not_exist=True)
                    if hasattr(scorer, 'add_string'):
                        scorer.add_string(target_str, detok_hypo_str)
                    else:
                        scorer.add(target_tokens, hypo_tokens)


            if args.knn_start > -1 and knn_num_samples_proc == args.knn_proc:
                break
            if args.save_knn_subset and total_saved >= args.save_knn_subset_num:
                break
            #if i > 10:
            #    break
        if args.knn_start > -1 and knn_num_samples_proc == args.knn_proc:
            break
        if args.save_knn_subset and total_saved >= args.save_knn_subset_num:
            break
        if args.knn_add_to_idx:
            adding_to_faiss += keys.shape[0]
            for fidx in range(len(args.trained_index)):
                faiss_indices[fidx].add_with_ids(keys, addids)
            #print(f"loop time {time.time()-knn_start_loop}s")

        #print(idx)
        #if idx == 0:
        #    break



        wps_meter.update(num_generated_tokens)
        progress.log({'wps': round(wps_meter.avg)})
        num_sentences += sample["nsentences"] if "nsentences" in sample else sample['id'].numel()

    if args.knn_q2gpu:
        index_ivf.quantizer = quantizer
        del quantizer_gpu

    if args.save_knn_dstore:
        if args.knn_start > -1:
            dstore_keys = dstore_keys[:total_saved]
            dstore_vals = dstore_vals[:total_saved]
            np.savez(args.dstore_mmap+".keys_vals.%d.%d" % (args.knn_start, args.knn_start + knn_num_samples_proc - 1), keys=dstore_keys, vals=dstore_vals)
            print("Final dstore position = %d" % (start_pos + total_saved - 1))
            print("Number of examples processed = %d" % knn_num_samples_proc)
            knn_samples_savefile = args.dstore_mmap+".samples.%d.%d" % (args.knn_start, args.knn_start + knn_num_samples_proc - 1)
        #else:
        #    knn_samples_savefile = args.dstore_mmap+".samples"
        #np.save(knn_samples_savefile, np.array(sample_order_lens, dtype=np.int))
        print("dstore_idx", dstore_idx, "final number of items added", num_items - skip, "total saved", total_saved)
        if not args.knn_add_to_idx:
            print("Keys", dstore_keys.shape, dstore_keys.dtype)
            print("Vals", dstore_vals.shape, dstore_vals.dtype)
        else:
            for widx, write_index in enumerate(args.write_index):
                faiss.write_index(faiss_indices[widx], write_index)
                print("Added to faiss", adding_to_faiss)
                #print("Final global position %d" % global_end)

    if args.knnmt and args.save_knns:
        pickle.dump(to_save_objects, open(args.save_knns_filename, "wb"))

    logger.info('NOTE: hypothesis and token scores are output in base 2')
    logger.info('Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if has_target and not args.score_reference:
        if args.bpe and not args.sacrebleu:
            if args.remove_bpe:
                logger.warning("BLEU score is being computed by splitting detokenized string on spaces, this is probably not what you want. Use --sacrebleu for standard 13a BLEU tokenization")
            else:
                logger.warning("If you are using BPE on the target side, the BLEU score is computed on BPE tokens, not on proper words.  Use --sacrebleu for standard 13a BLEU tokenization")
        # use print to be consistent with other main outputs: S-, H-, T-, D- and so on
        if args.target_lang == 'ja':
            print("Sending sacrebleu tokenier: ja-mecab")
            print(
                'Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string(tokenize='ja-mecab')),
                file=output_file)
        elif args.target_lang == 'zh':
            print("Sending sacrebleu tokenier: zh")
            print(
                'Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string(tokenize='zh')),
                file=output_file)
        else:
            print(
                'Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()),
                file=output_file)

    return scorer


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
