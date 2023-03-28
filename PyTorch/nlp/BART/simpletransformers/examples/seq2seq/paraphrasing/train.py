import os
import sys
import datetime
import logging
import time
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../")))
from simpletransformers.config.model_args import Seq2SeqArgs
from simpletransformers.seq2seq.seq2seq_model import Seq2SeqModel


from utils import load_data, clean_unnecessary_spaces
import argparse
import random
import hb_utils

try:
    from apex import amp
except ImportError:
    amp = None

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)


def     set_env_params():
    os.environ['PT_HPU_ENABLE_SYNC_OUTPUT_HOST'] = 'false'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--use_habana",
            action="store_true",
            help="Whether not to use Habana device when available"
        )
    parser.add_argument(
            "--lazy_mode",
            action="store_true",
            help="Enable lazy mode or not",
        )
    parser.add_argument(
            "--output_dir",
            default='/tmp/bart',
            type=str,
            help="Output dir",
        )
    parser.add_argument(
            "--no_cache",
            action="store_true",
            help="Whether not to cache data"
        )
    parser.add_argument(
            "--reprocess_input_data",
            action="store_true",
            help="Whether or not to reprocess input data"
        )
    parser.add_argument(
            "--no_cuda",
            action="store_true",
            help="Whether not to use CUDA when available"
        )
    parser.add_argument(
            "--use_fused_adam",
            action="store_true",
            help="Whether to use fused adamw on habana device"
        )
    parser.add_argument(
            "--use_fused_clip_norm",
            action="store_true",
            help="Whether to use fused clip norm on habana device"
        )
    parser.add_argument(
            "--local_rank",
            type=int,
            default=-1,
            help="local_rank for distributed training on gpus"
        )
    parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="random seed for initialization"
        )
    parser.add_argument(
            "--max_seq_length",
            type=int,
            default=128,
            help="maximum input sequence length"
        )
    parser.add_argument(
            "--train_batch_size",
            type=int,
            default=8,
            help="batch size for training"
        )
    parser.add_argument(
            "--fp16",
            action="store_true",
            help="Whether to use fp16"
        )
    parser.add_argument(
            "--bf16",
            type=str,
            help="Type of bf16 mixed precision implementation",
            choices=["none", "hmp", "autocast"]
        )
    parser.add_argument(
            "--hmp_bf16",
            default="ops_bf16_bart.txt",
            help="path to bf16 ops list in hmp O1 mode"
        )
    parser.add_argument(
            "--hmp_fp32",
            default="ops_fp32_bart.txt",
            help="path to fp32 ops list in hmp O1 mode"
        )
    parser.add_argument(
            "--hmp_opt_level",
            default="O1",
            help="choose optimization level for hmp"
        )
    parser.add_argument(
            "--hmp_verbose",
            action="store_true",
            help="enable verbose mode for hmp"
        )
    parser.add_argument(
            "--debug",
            action="store_true",
            help="Whether in debug mode"
        )
    parser.add_argument(
            "--save_steps",
            type=int,
            default=-1,
            help="number of steps to save the model"
        )
    parser.add_argument(
            "--max_steps",
            type=int,
            default=-1,
            help="number of maximum training steps"
        )
    parser.add_argument(
            "--save_optimizer_and_scheduler",
            action="store_true",
            help="Whether save optimizer and scheduler"
        )
    parser.add_argument(
            "--eval_batch_size",
            type=int,
            default=64,
            help="batch size for evaluation"
        )
    parser.add_argument(
            "--evaluate_during_training",
            action="store_true",
            help="Whether evaluate during training"
        )
    parser.add_argument(
            "--evaluate_during_training_steps",
            type=int,
            default=-1,
            help="evaluate every training steps"
        )
    parser.add_argument(
            "--evaluate_each_epoch",
            action="store_true",
            help="Whether evaluate after each epoch"
        )
    parser.add_argument(
            "--evaluate_generated_text",
            action="store_true",
            help="Whether evaluate the generated text"
        )
    parser.add_argument(
            "--save_model_every_epoch",
            action="store_true",
            help="Whether save the model after each epoch"
        )
    parser.add_argument(
            "--save_eval_checkpoints",
            action="store_true",
            help="Whether save the checkpoint after evaluation"
        )
    parser.add_argument(
            "--save_best_model",
            action="store_true",
            help="Whether save the best model"
        )
    parser.add_argument(
            "--logging_steps",
            type=int,
            default=50,
            help="number of logging steps"
        )
    parser.add_argument(
            "--num_train_epochs",
            type=int,
            default=3,
            help="number of epochs for training"
        )
    parser.add_argument(
            "--num_return_sequences",
            type=int,
            default=1,
            help="number of return sequences during beam sampling"
        )
    parser.add_argument(
            "--predict",
            action="store_true",
            help="Whether generate text given input"
        )
    #################### distributed training ######################
    parser.add_argument(
            '--dl_worker_type',
            default='HABANA',
            type=lambda x: x.upper(),
            choices = ["MT", "MP", "HABANA"],
            help='select multithreading or multiprocessing'
        )
    parser.add_argument(
            '--world_size',
            default=1,
            type=int,
            metavar='N',
            help='number of total workers (default: 1)'
        )
    parser.add_argument(
            '--process_per_node',
            default=8,
            type=int,
            metavar='N',
            help='Number of process per node'
        )
    parser.add_argument(
            '--distributed',
            action='store_true',
            help='whether to enable distributed mode and run on multiple devices'
        )
    parser.add_argument(
            '--dist_url',
            default='env://',
            help='url used to set up distributed training'
        )
    parser.add_argument(
        "--data_dir",
        default="",
        type=str,
        help="The input data dir. If no data dir, will run with ./data under local directory.",
    )
    args = parser.parse_args()

    model_args = Seq2SeqArgs()
    model_args.debug = True if args.debug else False
    model_args.eval_batch_size = args.eval_batch_size
    model_args.evaluate_during_training = True if args.evaluate_during_training else False
    model_args.evaluate_during_training_steps = args.evaluate_during_training_steps
    model_args.evaluate_each_epoch = True if args.evaluate_each_epoch else False
    model_args.evaluate_during_training_verbose = True
    model_args.evaluate_generated_text = True if args.evaluate_generated_text else False
    model_args.fp16 = True if args.fp16 else False
    model_args.bf16 = args.bf16
    model_args.hmp_bf16 = args.hmp_bf16
    model_args.hmp_fp32 = args.hmp_fp32
    model_args.hmp_opt_level = args.hmp_opt_level
    model_args.hmp_verbose = True if args.hmp_verbose else False
    model_args.learning_rate = 5e-5
    model_args.gradient_accumulation_steps = 1
    model_args.max_seq_length = args.max_seq_length
    model_args.num_train_epochs = args.num_train_epochs
    model_args.overwrite_output_dir = True
    model_args.reprocess_input_data = True if args.reprocess_input_data else False
    model_args.logging_steps = args.logging_steps
    model_args.save_eval_checkpoints = True if args.save_eval_checkpoints else False
    model_args.save_steps = args.save_steps
    model_args.save_model_every_epoch = True if args.save_model_every_epoch else False
    model_args.save_best_model = True if args.save_best_model else False
    model_args.save_optimizer_and_scheduler = True if args.save_optimizer_and_scheduler else False
    model_args.train_batch_size = args.train_batch_size
    model_args.use_multiprocessing = False
    model_args.use_multiprocessing_for_evaluation = False
    model_args.predict = True if args.predict else False
    model_args.do_sample = True
    model_args.num_beams = None
    model_args.num_return_sequences = args.num_return_sequences
    model_args.max_length = args.max_seq_length
    model_args.top_k = 50
    model_args.top_p = 0.95

    model_args.max_steps = args.max_steps
    model_args.seed = args.seed
    model_args.use_habana = args.use_habana
    model_args.use_fused_adam = args.use_fused_adam
    model_args.use_fused_clip_norm = args.use_fused_clip_norm
    model_args.output_dir = args.output_dir
    model_args.best_model_dir = args.output_dir
    model_args.tensorboard_dir = args.output_dir
    model_args.no_cache = True if args.no_cache else False
    model_args.cache_dir = args.output_dir

    if args.use_habana and args.use_fused_adam:
        model_args.optimizer = 'FusedAdamW'
        model_args.max_grad_norm = 1.0
    else:
        model_args.optimizer = 'AdamW'
        model_args.adafactor_relative_step = False
        model_args.adafactor_scale_parameter = False
        model_args.adafactor_warmup_init = False

    model_args.scheduler = "linear_schedule_with_warmup"
    return args, model_args

def load_train_val_data():
    if args.local_rank not in [-1, 0]:
        if args.use_habana:
            hb_utils.barrier()
        else:
            torch.distributed.barrier()

    if args.local_rank in [-1, 0]:
        # Google Data
        train_df = pd.read_csv(os.path.join(args.data_dir,"data/train.tsv"), sep="\t").astype(str)
        eval_df = pd.read_csv(os.path.join(args.data_dir,"data/dev.tsv"), sep="\t").astype(str)

        train_df = train_df.loc[train_df["label"] == "1"]
        eval_df = eval_df.loc[eval_df["label"] == "1"]

        train_df = train_df.rename(
            columns={"sentence1": "input_text", "sentence2": "target_text"}
        )
        eval_df = eval_df.rename(
            columns={"sentence1": "input_text", "sentence2": "target_text"}
        )

        train_df = train_df[["input_text", "target_text"]]
        eval_df = eval_df[["input_text", "target_text"]]

        train_df["prefix"] = "paraphrase"
        eval_df["prefix"] = "paraphrase"

        # MSRP Data
        '''
        train_df = pd.concat(
            [
                train_df,
                load_data("data/msr_paraphrase_train.txt", "Quality", "#1_String", "#2_String"),
            ]
        )
        eval_df = pd.concat(
            [
                eval_df,
                load_data("data/msr_paraphrase_test.txt", "#1 String", "#2 String", "Quality"),
            ]
        )
        '''
    train_df = []
    eval_df = []

    # Quora Data

    # The Quora Dataset is not separated into train/test, so we do it manually the first time.
    if not os.path.exists("data/quora_train.tsv") or not os.path.exists("data/quora_test.tsv"):
        df = load_data(
            os.path.join(args.data_dir, "data/quora_duplicate_questions.tsv"), "question1", "question2", "is_duplicate"
        )
        q_train, q_test = train_test_split(df)
        print('Splitting train and test...')
        q_train.to_csv(os.path.join(args.data_dir, "data/quora_train.tsv"), sep="\t")
        q_test.to_csv(os.path.join(args.data_dir, "data/quora_test.tsv"), sep="\t")
    else:
        # The code block above only needs to be run once.
        # After that, the two lines below are sufficient to load the Quora dataset.
        print('Reading train and test...')
        q_train = pd.read_csv("data/quora_train.tsv", sep="\t")
        q_test = pd.read_csv("data/quora_test.tsv", sep="\t")

    train_df = q_train #pd.concat([train_df, q_train])
    eval_df = q_test #pd.concat([eval_df, q_test])

    train_df = train_df[["prefix", "input_text", "target_text"]]
    eval_df = eval_df[["prefix", "input_text", "target_text"]]

    train_df = train_df.dropna()
    eval_df = eval_df.dropna()

    train_df["input_text"] = train_df["input_text"].apply(clean_unnecessary_spaces)
    train_df["target_text"] = train_df["target_text"].apply(clean_unnecessary_spaces)

    eval_df["input_text"] = eval_df["input_text"].apply(clean_unnecessary_spaces)
    eval_df["target_text"] = eval_df["target_text"].apply(clean_unnecessary_spaces)


    if args.local_rank == 0:
        if args.use_habana:
            hb_utils.barrier()
        else:
            torch.distributed.barrier()
    return train_df, eval_df


def main(args, model_args):
    if args.dl_worker_type == "MP":
        try:
            # Default 'fork' doesn't work with synapse. Use 'forkserver' or 'spawn'
            torch.multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass
    elif args.dl_worker_type == "HABANA":
        try:
            import habana_dataloader
        except ImportError:
            assert False, "Could Not import habana dataloader package"

    #if args.apex:
    #    if sys.version_info < (3, 0):
    #        raise RuntimeError("Apex currently only supports Python 3. Aborting.")
    #    if amp is None:
    #        raise RuntimeError("Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
    #                           "to enable mixed-precision training.")
    hb_utils.init_distributed_mode(args)
    if hasattr(args, "rank"):
        args.local_rank = args.rank
    print('####################### These are args: ######################')
    print(args)

    model_args.dl_worker_type = args.dl_worker_type
    model_args.world_size = args.world_size
    model_args.process_per_node = args.process_per_node
    model_args.distributed = args.distributed
    model_args.dist_url = args.dist_url

    args.is_master = False
    if args.local_rank in [-1, 0]:
        args.is_master = True
    model_args.is_master = args.is_master
    model_args.local_rank = args.local_rank
    print("############### local_rank is_master #############", model_args.local_rank, model_args.is_master)

    if not args.lazy_mode:
        print('######### Eager Mode ########')
        os.environ["PT_HPU_LAZY_MODE"] = "2"
    else:
        print('######### Lazy Mode ########')
    model_args.lazy_mode = args.lazy_mode


    if model_args.use_habana is True:
        device = torch.device("hpu")
        args.n_gpu = 0
        print("########## HPU ##########")

    if args.no_cuda is False:
        if torch.cuda.is_available():
            n_gpu = torch.cuda.device_count()
            if n_gpu > 1:
                torch.cuda.set_device(args.local_rank)
                device = torch.device("cuda", args.local_rank)
            else:
                device = torch.device("cuda")
            args.n_gpu = n_gpu
            print("########## GPU n_gpu ##########", args.n_gpu)
        else:
            device = torch.device("cpu")
            args.n_gpu = 0
            print("########## CPU ##########")

    model_args.device = device
    model_args.n_gpu = args.n_gpu

    #if args.deterministic:
    #    seed = args.seed
    #    random.seed(seed)
    #    if args.device == 'cuda':
    #        torch.cuda.manual_seed(seed)
    #else:
    #    seed = None


    train_df, eval_df = load_train_val_data()

    if model_args.device == 'hpu' and model_args.workers > 0:
        # patch torch cuda functions that are being unconditionally invoked
        # in the multiprocessing data loader
        torch.cuda.current_device = lambda: None
        torch.cuda.set_device = lambda x: None

    model = Seq2SeqModel(
        encoder_decoder_type="bart",
        encoder_decoder_name="facebook/bart-base",
        args=model_args,
        use_cuda=True if args.n_gpu > 0 else False,
        cuda_device=args.local_rank if args.n_gpu > 0 else -1,
    )

    start_time = time.time()

    model.train_model(train_df, eval_data=eval_df, output_dir=args.output_dir)

    ####################### prediction #######################
    if args.predict and args.local_rank in [-1, 0]:
        to_predict = [
            prefix + ": " + str(input_text)
            for prefix, input_text in zip(
                eval_df["prefix"].tolist(), eval_df["input_text"].tolist()
            )
        ]
        truth = eval_df["target_text"].tolist()

        print("Start testing")
        start_time = time.time()
        #
        preds = model.predict(to_predict)
        #
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Testing time {}'.format(total_time_str))

        os.makedirs(os.path.join(args.output_dir, "predictions"), exist_ok=True)
        pred_time = f"_{datetime.datetime.now()}"
        pred_text = os.path.join(args.output_dir, "predictions", "pred_text"+pred_time+".txt")

        with open(pred_text, "w") as f:
            for i, text in enumerate(eval_df["input_text"].tolist()):
                f.write(str(text) + "\n\n")

                f.write("Truth:\n")
                f.write(truth[i] + "\n\n")

                f.write("Prediction:\n")
                for pred in preds[i]:
                    f.write(str(pred) + "\n")
                f.write(
                    "________________________________________________________________________________\n"
                )

        results = model.compute_metrics(
                    truth, preds
                    )
        print('Prediction results:')
        print(results)

        pred_results = os.path.join(args.output_dir, "predictions", "pred_results"+pred_time+".csv")
        report = pd.DataFrame(results, index=[0])
        report.to_csv(
                    pred_results,
                    index=False,
            )
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total time {}'.format(total_time_str))

if __name__ == "__main__":
    # set_env_params()
    args, model_args = parse_args()
    main(args, model_args)
