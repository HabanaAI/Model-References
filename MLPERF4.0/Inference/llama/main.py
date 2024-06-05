###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

import argparse
import os
import logging
import sys
import time
import mlperf_loadgen as lg
import SUT

sys.path.insert(0, os.getcwd())

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-70B-MAIN")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str,
                        choices=["Offline", "Server"], default="Offline", help="Scenario")
    parser.add_argument("--sut-server", type=str,
                        default="http://localhost:8080", help="Address of the TGI server")
    parser.add_argument("--dataset-path", type=str,
                        default="/mnt/weka/data/mlperf_inference/llama2/processed-data.pkl")
    parser.add_argument("--accuracy", action="store_true",
                        help="Run accuracy mode")
    parser.add_argument("--audit-conf", type=str, default="audit.conf",
                        help="audit config for LoadGen settings during compliance runs")
    parser.add_argument("--mlperf-conf", type=str,
                        default="mlperf.conf", help="mlperf rules config")
    parser.add_argument("--user-conf", type=str, default="configs/fp8.conf",
                        help="user config for user LoadGen settings such as target QPS")
    parser.add_argument("--total-sample-count", type=int, default=24576)
    parser.add_argument("--output-log-dir", type=str,
                        default="build/logs", help="Where logs are saved")
    parser.add_argument("--enable-log-trace", action="store_true",
                        help="Enable log tracing. This file can become quite large")
    parser.add_argument("--max-num-threads", type=int, default=1024,
                        help="Max number of concurrent issue_query threads")

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    settings = lg.TestSettings()
    settings.FromConfig(args.mlperf_conf, "llama2-70b", args.scenario)
    settings.FromConfig(args.user_conf, "llama2-70b", args.scenario)

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    os.makedirs(args.output_log_dir, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = args.output_log_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = args.enable_log_trace

    if args.scenario == "Server":
        sut = SUT.Server(args)
        settings.scenario = lg.TestScenario.Server
    else:
        sut = SUT.Offline(args)
        settings.scenario = lg.TestScenario.Offline

    lgSUT = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)
    log.info("Starting Benchmark run")
    t_start = time.time()
    lg.StartTestWithLogSettings(
        lgSUT, sut.qsl, settings, log_settings, args.audit_conf)
    t_end = time.time()

    gen_tokens = sum(sut.gen_tok_lens)
    duration = t_end - t_start
    if args.accuracy and args.scenario == "Offline":
        log.info("Estimated performance for accuracy run is {:.1f} tokens per second".format(
            gen_tokens/duration))
    log.info("Test took {:.1f} sec : generated {} tokens and processed {} queries".format(
        duration, gen_tokens, len(sut.gen_tok_lens)))

    log.info("Run Completed!")

    log.info("Destroying SUT...")
    lg.DestroySUT(lgSUT)

    log.info("Destroying QSL...")
    lg.DestroyQSL(sut.qsl)


if __name__ == "__main__":
    main()
