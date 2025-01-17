# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import random
import argparse
import numpy as np

from src.slurm import init_signal_handler, init_distributed_mode
from src.data.loader import check_data_params, load_data
from src.utils import bool_flag, initialize_exp, set_sampling_probs, shuf_order
from src.model import check_model_params, build_model
from src.model import build_model_multilang
from src.model.memory import HashingMemory
from src.trainer import SingleTrainer, EncDecTrainer
from src.evaluation.evaluator import SingleEvaluator, EncDecEvaluator
import torch


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="./dumped/",
                        help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="",
                        help="Experiment name")
    parser.add_argument("--save_periodic", type=int, default=0,
                        help="Save the model periodically (0 to disable)")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")

    # float16 / AMP API
    parser.add_argument("--fp16", type=bool_flag, default=False,
                        help="Run model with float16")
    parser.add_argument("--amp", type=int, default=-1,
                        help="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.")

    # only use an encoder (use a specific decoder for machine translation)
    parser.add_argument("--encoder_only", type=bool_flag, default=True,
                        help="Only use an encoder")

    # model parameters
    parser.add_argument("--emb_dim", type=int, default=512,
                        help="Embedding layer size")
    parser.add_argument("--n_layers", type=int, default=4,
                        help="Number of Transformer layers")
    parser.add_argument("--share_enc", type=int, default=-1,
                        help="Number of Transformer layers")
    parser.add_argument("--share_dec", type=int, default=-1,
                        help="Number of Transformer layers")

    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of Transformer heads")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout")
    parser.add_argument("--attention_dropout", type=float, default=0,
                        help="Dropout in the attention layer")
    parser.add_argument("--gelu_activation", type=bool_flag, default=False,
                        help="Use a GELU activation instead of ReLU")
    parser.add_argument("--share_inout_emb", type=bool_flag, default=True,
                        help="Share input and output embeddings")
    parser.add_argument("--sinusoidal_embeddings", type=bool_flag, default=False,
                        help="Use sinusoidal embeddings")
    parser.add_argument("--use_lang_emb", type=bool_flag, default=True,
                        help="Use language embedding")

    # memory parameters
    parser.add_argument("--use_memory", type=bool_flag, default=False,
                        help="Use an external memory")
    if parser.parse_known_args()[0].use_memory:
        HashingMemory.register_args(parser)
        parser.add_argument("--mem_enc_positions", type=str, default="",
                            help="Memory positions in the encoder ('4' for inside layer 4, '7,10+' for inside layer 7 and after layer 10)")
        parser.add_argument("--mem_dec_positions", type=str, default="",
                            help="Memory positions in the decoder. Same syntax as `mem_enc_positions`.")

    # adaptive softmax
    parser.add_argument("--asm", type=bool_flag, default=False,
                        help="Use adaptive softmax")
    if parser.parse_known_args()[0].asm:
        parser.add_argument("--asm_cutoffs", type=str, default="8000,20000",
                            help="Adaptive softmax cutoffs")
        parser.add_argument("--asm_div_value", type=float, default=4,
                            help="Adaptive softmax cluster sizes ratio")

    # causal language modeling task parameters
    parser.add_argument("--context_size", type=int, default=0,
                        help="Context size (0 means that the first elements in sequences won't have any context)")

    # masked language modeling task parameters
    parser.add_argument("--word_pred", type=float, default=0.15,
                        help="Fraction of words for which we need to make a prediction")
    parser.add_argument("--sample_alpha", type=float, default=0,
                        help="Exponent for transforming word counts to probabilities (~word2vec sampling)")
    parser.add_argument("--word_mask_keep_rand", type=str, default="0.8,0.1,0.1",
                        help="Fraction of words to mask out / keep / randomize, among the words to predict")

    # input sentence noise
    parser.add_argument("--word_shuffle", type=float, default=0,
                        help="Randomly shuffle input words (0 to disable)")
    parser.add_argument("--word_dropout", type=float, default=0,
                        help="Randomly dropout input words (0 to disable)")
    parser.add_argument("--word_blank", type=float, default=0,
                        help="Randomly blank input words (0 to disable)")

    # ED-MLM Parameters
    parser.add_argument("--edmlm_full", type=bool_flag, default=False,
                        help="Full prediction at decoder")

    # data
    parser.add_argument("--data_path", type=str, default="",
                        help="Data path")
    parser.add_argument("--lgs", type=str, default="",
                        help="Languages (lg1-lg2-lg3 .. ex: en-fr-es-de)")
    parser.add_argument("--max_vocab", type=int, default=-1,
                        help="Maximum vocabulary size (-1 to disable)")
    parser.add_argument("--min_count", type=int, default=0,
                        help="Minimum vocabulary count")
    parser.add_argument("--lg_sampling_factor", type=float, default=-1,
                        help="Language sampling factor")

    # batch parameters
    parser.add_argument("--bptt", type=int, default=256,
                        help="Sequence length")
    parser.add_argument("--max_len", type=int, default=100,
                        help="Maximum length of sentences (after BPE)")
    parser.add_argument("--group_by_size", type=bool_flag, default=True,
                        help="Sort sentences by size during the training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of sentences per batch")
    parser.add_argument("--max_batch_size", type=int, default=0,
                        help="Maximum number of sentences per batch (used in combination with tokens_per_batch, 0 to disable)")
    parser.add_argument("--tokens_per_batch", type=int, default=-1,
                        help="Number of tokens per batch")

    # training parameters
    parser.add_argument("--split_data", type=bool_flag, default=False,
                        help="Split data across workers of a same node")
    parser.add_argument("--optimizer", type=str, default="adam,lr=0.0001",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--epoch_size", type=int, default=100000,
                        help="Epoch size / evaluation frequency (-1 for parallel data size)")
    parser.add_argument("--max_epoch", type=int, default=100000,
                        help="Maximum epoch size")
    parser.add_argument("--stopping_criterion", type=str, default="",
                        help="Stopping criterion, and number of non-increase before stopping the experiment")
    parser.add_argument("--validation_metrics", type=str, default="",
                        help="Validation metrics")
    parser.add_argument("--accumulate_gradients", type=int, default=1,
                        help="Accumulate model gradients over N iterations (N times larger batch sizes)")

    # training coefficients
    parser.add_argument("--lambda_edmlm", type=str, default="1",
                        help="Prediction coefficient (EDMLM)")
    parser.add_argument("--lambda_mlm", type=str, default="1",
                        help="Prediction coefficient (MLM)")
    parser.add_argument("--lambda_clm", type=str, default="1",
                        help="Causal coefficient (LM)")
    parser.add_argument("--lambda_pc", type=str, default="1",
                        help="PC coefficient")
    parser.add_argument("--lambda_ae", type=str, default="1",
                        help="AE coefficient")
    parser.add_argument("--lambda_mt", type=str, default="1",
                        help="MT coefficient")
    parser.add_argument("--lambda_bt", type=str, default="1",
                        help="BT coefficient")
    # parser.add_argument("--lambda_bt_otf", type=str, default="0",
    #                     help="BT coefficient on the fly separate")
    parser.add_argument("--bt_sync", type=int, default=1,
                        help="log_per_iter")


    # training steps
    parser.add_argument("--clm_steps", type=str, default="",
                        help="Causal prediction steps (CLM)")
    parser.add_argument("--mlm_steps", type=str, default="",
                        help="Masked prediction steps (MLM / TLM)")
    parser.add_argument("--edmlm_steps", type=str, default="",
                        help="Masked prediction steps (EDMLM / EDTLM)")
    parser.add_argument("--mt_steps", type=str, default="",
                        help="Machine translation steps")
    parser.add_argument("--ae_steps", type=str, default="",
                        help="Denoising auto-encoder steps")
    parser.add_argument("--bt_steps", type=str, default="",
                        help="Back-translation steps")
    parser.add_argument("--pc_steps", type=str, default="",
                        help="Parallel classification steps")

    parser.add_argument("--encoder_bt_steps", type=str, default="",
                        help="Encoder Back-translation steps")
    parser.add_argument("--qbt_steps", type=str, default="",
                        help="Quick Back-translation steps")
    
    
    # logging
    parser.add_argument("--log_per_iter", type=int, default=1,
                        help="log_per_iter")

    # reload pretrained embeddings / pretrained model / checkpoint
    parser.add_argument("--reload_emb", type=str, default="",
                        help="Reload pretrained word embeddings")
    parser.add_argument("--reload_model", type=str, default="",
                        help="Reload a pretrained model")
    parser.add_argument("--reload_checkpoint", type=str, default="",
                        help="Reload a checkpoint")

    parser.add_argument("--reload_2nd_model", type=str, default="",
                        help="Reload a secondary parallel model. X ->(m1) Y1 =>(m2) X2")

    # beam search (for MT only)
    parser.add_argument("--early_stopping", type=bool_flag, default=False,
                        help="Early stopping, stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.")

    # evaluation
    parser.add_argument("--eval_bleu", type=bool_flag, default=False,
                        help="Evaluate BLEU score during MT training")
    parser.add_argument("--eval_only", type=bool_flag, default=False,
                        help="Only run evaluations")
    parser.add_argument("--infer_train", type=bool_flag, default=False,
                        help="Infer training data")

    # debug
    parser.add_argument("--debug_train", type=bool_flag, default=False,
                        help="Use valid sets for train sets (faster loading)")
    parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                        help="Debug multi-GPU / multi-node within a SLURM job")
    parser.add_argument("--debug", help="Enable all debug flags",
                        action="store_true")

    # multi-gpu / multi-node
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--master_port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")

    # seed
    parser.add_argument("--seed", type=int, default=-1, help="If >= 0, set the seed")

    
     # beam search (for MT only)
    parser.add_argument("--beam_size", type=int, default=5,
                        help="Beam size, default = 1 (greedy decoding)")
    parser.add_argument("--length_penalty", type=float, default=0.6,
                        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.")

    parser.add_argument("--nbest", type=int, default=3,
                        help="Beam size, default = 1 (greedy decoding)")
    parser.add_argument("--sampling_topp", type=float, default=-1,
                        help="sampling_topp.")

    parser.add_argument("--fast_beam", type=bool_flag, default=True,
                        help="fast_beam")
    parser.add_argument("--fast_beam_epoch", type=int, default=1,
                        help="fast_beam_epoch")
    parser.add_argument("--mbeam_size", type=int, default=-1,
                        help="mbeam_size")
    
    
    
    
    return parser

def main(params):

    # initialize the multi-GPU / multi-node training
    init_distributed_mode(params)

    if params.infer_train:
        log_filename = 'infer.train.log'
        params_filename = 'infer.train.params.pkl'
    else:
        log_filename, params_filename = None, None

    # initialize the experiment
    logger = initialize_exp(params, log_filename, params_filename)

    # initialize SLURM signal handler for time limit / pre-emption
    init_signal_handler()

    # load data
    data = load_data(params)
    logger.info('INIT MODEL HERE')
    logger.info(data)

    # build model
    if params.encoder_only:
        model = build_model(params, data['dico'])
    else:
        try:
            build_func = build_model_multilang if (params.share_enc > -1 or params.share_dec > -1) else build_model
            logger.info('Build function: {}'.format(build_func.__name__))
            # encoder, decoder = build_model(params, data['dico'])
            encoder, decoder = build_func(params, data['dico'])
        except Exception as e:
            # print(data)
            raise e

    # build trainer, reload potential checkpoints / build evaluator
    if params.encoder_only:
        trainer = SingleTrainer(model, data, params)
        evaluator = SingleEvaluator(trainer, data, params)
    else:
        trainer = EncDecTrainer(encoder, decoder, data, params)
        evaluator = EncDecEvaluator(trainer, data, params)

    # evaluation
    if params.eval_only:
        scores = evaluator.run_all_evals(trainer)
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))
        exit()

    if params.infer_train:
        print('**** Infer Train')
        logger.info('======== Start Generating ===========')
        assert isinstance(evaluator, EncDecEvaluator)
        scores = evaluator.infer_train(trainer)
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))
        exit()

    # set sampling probabilities for training
    set_sampling_probs(data, params)

    # language model training
    for _ in range(params.max_epoch):

        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        trainer.n_sentences = 0

        while trainer.n_sentences < trainer.epoch_size:

            # CLM steps
            for lang1, lang2 in shuf_order(params.clm_steps, params):
                trainer.clm_step(lang1, lang2, params.lambda_clm)

            # MLM steps (also includes TLM if lang2 is not None)
            for lang1, lang2 in shuf_order(params.mlm_steps, params):
                trainer.mlm_step(lang1, lang2, params.lambda_mlm)

            # parallel classification steps
            for lang1, lang2 in shuf_order(params.pc_steps, params):
                trainer.pc_step(lang1, lang2, params.lambda_pc)

            # denoising auto-encoder steps
            for lang in shuf_order(params.ae_steps):
                trainer.mt_step(lang, lang, params.lambda_ae)

            # machine translation steps
            for lang1, lang2 in shuf_order(params.mt_steps, params):
                trainer.mt_step(lang1, lang2, params.lambda_mt)

                
            if (trainer.n_sentences % 1 == 0): #not random_mono: #
                for lang1, lang2, lang3 in shuf_order(params.encoder_bt_steps):
                    trainer.encoder_bt_step(lang1, lang2, lang3, params.lambda_bt)

            if (trainer.n_sentences % 1 == 0):
                for lang1, lang2, lang3 in shuf_order(params.qbt_steps):
                    trainer.qbt_step(lang1, lang2, lang3, params.lambda_bt)   
                
             
            # back-translation steps
            if params.bt_sync == 1:
                for lang1, lang2, lang3 in shuf_order(params.bt_steps):
                    trainer.bt_step(lang1, lang2, lang3, params.lambda_bt)
            else:
                assert params.bt_sync > 1
                if trainer.n_iter % params.bt_sync == 0 and trainer.n_iter > 0:
                    trainer.update_syn_model()
                for lang1, lang2, lang3 in shuf_order(params.bt_steps):
                    trainer.bt_sync_step(lang1, lang2, lang3, params.lambda_bt)

                # # AE - MT training (on the fly back-translation)
                # # start on-the-fly batch generations
                # if not getattr(params, 'started_otf_batch_gen', False):
                #     otf_iterator = trainer.otf_bt_gen_async()
                #     params.started_otf_batch_gen = True
                # # update model parameters on subprocesses
                # if trainer.n_iter % params.otf_sync_params_every == 0:
                #     trainer.otf_sync_params()
                # # get training batch from CPU
                # before_gen = time.time()
                # batches = next(otf_iterator)
                # trainer.gen_time += time.time() - before_gen
                # # training
                # for batch in batches:
                #     lang1, lang2, lang3 = batch['lang1'], batch['lang2'], batch['lang3']
                #     # 2-lang back-translation - autoencoding
                #     if lang1 != lang2 == lang3:
                #         trainer.otf_bt(batch, params.lambda_xe_otfa, params.otf_backprop_temperature)
                #     # 2-lang back-translation - parallel data
                #     elif lang1 == lang3 != lang2:
                #         trainer.otf_bt(batch, params.lambda_xe_otfd, params.otf_backprop_temperature)
                #     # 3-lang back-translation - parallel data
                #     elif lang1 != lang2 and lang2 != lang3 and lang1 != lang3:
                #         trainer.otf_bt(batch, params.lambda_xe_otfd, params.otf_backprop_temperature)

            trainer.iter()

        logger.info("============ End of epoch %i ============" % trainer.epoch)

        # evaluate perplexity
        scores = evaluator.run_all_evals(trainer)

        # print / JSON log
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        if params.is_master:
            logger.info("__log__:%s" % json.dumps(scores))

        # end of epoch
        trainer.save_best_model(scores)
        trainer.save_periodic()
        trainer.end_epoch(scores)


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    if params.seed >= 0:
        print('| Set seed {}'.format(params.seed))
        torch.manual_seed(params.seed)
        torch.cuda.manual_seed(params.seed)
        random.seed(params.seed)
        np.random.seed(params.seed)

    # debug mode
    if params.debug:
        params.exp_name = 'debug'
        params.exp_id = 'debug_%08i' % random.randint(0, 100000000)
        params.debug_slurm = True
        params.debug_train = True

    # check parameters
    check_data_params(params)
    check_model_params(params)

    # run experiment
    main(params)
