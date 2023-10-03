###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
os.environ.setdefault('PT_HPU_INFERENCE_MODE', '1')

import argparse
import mlperf_loadgen as lg
import sys

from hgu_options import get_options_dict
import habana_generation_utils as hgu

sys.path.insert(0, os.getcwd())

scenario_map = {
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["Offline", "Server"], default="Offline", help="Scenario")
    parser.add_argument("--model-path", default="/mnt/weka/data/pytorch/gpt-j", help="")
    parser.add_argument("--dataset-path", default="/mnt/weka/data/pytorch/gpt-j/cnn_eval.json", help="")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument("--dtype", choices=["bfloat16", "float32", "float8"], default="bfloat16",
                        help="data type of the model, choose from bfloat16, float32 and float8")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "hpu", "socket"],
                        default="hpu", help="device to run the inference on")
    parser.add_argument("--mlperf_conf", default="mlperf.conf", help="mlperf rules config")
    parser.add_argument("--user_conf", default="user.conf",
                        help="user config for user LoadGen settings such as target QPS")
    parser.add_argument("--max_examples", type=int, default=13368,
                        help="Maximum number of examples to consider (not limited by default)")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--quantization_file", "-qf", type=str,
                        help="Read quantization configuration from a file")
    parser.add_argument("--log_path", default="build/logs")
    parser.add_argument("--options", type=str, default='',
                        help="Coma-seperated list of options used in generation")
    parser.add_argument("--profile", action='store_true', help="Enable profiling")
    parser.add_argument("--profile_type", type=str, choices=["tb", "hltv"], default='tb', help="Profiling format")
    parser.add_argument("--profile_tokens", type=int, default=5, help="Number of tokens to profile")
    parser.add_argument("--help_options", action="store_true", help="Show detailed option help")
    parser.add_argument("--fake_device", action='store_true', help="Enable dummy device with estimated delay")
    parser.add_argument("--fake_dataset", action='store_true', help="Enable dummy dataset")
    parser.add_argument("--stdout", action="store_true", help="Print logs to stdout instead of a file")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.num_workers != 1:
        assert args.device != 'hpu', "In order to run more than 1 worker, you need to set device to 'socket'"
    if args.help_options is True:
        print(hgu.generate_option_help())
        sys.exit(0)

    if args.scenario == "Offline":
        if args.device == "socket":
            from socket_backend import SUT_Offline
            sut = SUT_Offline(args)
        else:
            from backend import SUT_Offline
            options = get_options_dict(args.options)
            sut = SUT_Offline(args, options)
    else:
        if args.device == "socket":
            from socket_backend import SUT_Server
            sut = SUT_Server(args)
        else:
            from backend import SUT_Server
            options = get_options_dict(args.options)
            sut = SUT_Server(args, options)

    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    # Need to update the conf
    settings.FromConfig(args.mlperf_conf, "gptj", args.scenario)
    settings.FromConfig(args.user_conf, "gptj", args.scenario)

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly
    os.makedirs(args.log_path, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = args.log_path
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = True

    lg.StartTestWithLogSettings(sut.sut, sut.qsl, settings, log_settings)

    print("Test Done!")

    print("Destroying SUT...")
    sut.close_log_file()
    lg.DestroySUT(sut.sut)

    print("Destroying QSL...")
    lg.DestroyQSL(sut.qsl)


if __name__ == "__main__":
    main()
