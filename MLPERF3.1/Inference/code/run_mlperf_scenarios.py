###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################
import argparse
import yaml
import typing
import subprocess
import logging
import sys
import json
import shutil
import re
import os
from pathlib import Path
import time

scenarios_config = yaml.full_load(open("scenarios.yaml"))
logging.basicConfig(level=logging.INFO)
modes = ["Server", "Offline"]
units_map = {"Server": "Queries/s", "Offline": "Samples/s"}


def get_configuration(scenarios) -> typing.Tuple[str, str]:
    runs = []
    for scenario in scenarios:
        if scenario in scenarios_config["scenarios"]:
            for mode in modes:
                runs.append((scenario, mode))
        else:
            try:
                scenario, mode = scenario.split("_")
                assert mode in modes
                runs.append((scenario, mode))
            except:
                logging.error(
                    f"Scenario {scenario} not supported, see scenarios.yaml to view supported scenarios"
                )
                exit()
    return runs


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scenarios",
        nargs="+",
        help="List of scenarios e.g. gpt-j_Server or gpt-j separated by space, to run all possible scenarios set first element to 'all'",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        default="./results",
        help="Path to save results folder in",
    )
    parser.add_argument(
        "--mlperf-path", help="Path to mlperf inference directory"
    )
    parser.add_argument("--mode", type=str, choices=["full", "perf", "acc"], default="full", help="dev options to shorten test time")
    args = parser.parse_args()
    return args


def run_inference(base_dir, command, mode, accuracy, scenario):
    command += f" --scenario {mode}"
    if accuracy:
        command += " --accuracy"
    logging.info(command)
    try:
        subprocess.run(command, check=True, shell=True, cwd=base_dir)
    except subprocess.CalledProcessError as e:
        sys.exit(f"Failed running {scenario}_{mode}")


def evaluate(base_dir):
    start_time = time.time()
    # Assuming script naming convention is consistent between models
    command = "python evaluation.py | tee -a ./build/logs/accuracy.txt"
    logging.info(command)
    try:
        subprocess.run(command, check=True, shell=True, cwd=base_dir)
    except subprocess.CalledProcessError as e:
        sys.exit(f"Failed evaluating {base_dir}")
    return time.time() - start_time


def verify_thresholds(benchmark, results: typing.Dict[str, typing.Any]):
    error = ""
    valid = True
    thresholds = scenarios_config["benchmarks"][benchmark]
    for metric, threshold in thresholds.items():
        if results[metric] < threshold:
            error += f"{metric} "
            valid = False
    results["valid"] = valid
    results["error"] = error
    return results


def get_results(accuracy_path, benchmark):
    text = open(accuracy_path / "accuracy.txt").readlines()
    results = None
    for line in text:
        object_results = re.match("(\{.*?\})", line)
        if object_results is not None:
            results = yaml.full_load(object_results.group(1))
    if results is None:
        return sys.exit(f"No metrics found for {benchmark}")
    results = verify_thresholds(benchmark, results)
    return results


def get_performance(performance_path, mode):
    perf = {}
    text = open(performance_path / "mlperf_log_summary.txt").read()
    perf_pattern = (
        "Samples per second: (.+?)\n"
        if mode == "Offline"
        else "Scheduled samples per second : (.+?)\n"
    )
    validity_pattern = "Result is : (.+?)\n"
    perf['samples_per_seconds'] = re.search(perf_pattern, text).group(1)
    perf['validity'] = re.search(validity_pattern, text).group(1)

    return perf

def verify_performance(perf_validity, results: typing.Dict[str, typing.Any]):
    if perf_validity == "INVALID":
        results["valid"] = False
        results["error"] = "invalid"
    return results

def write_summary(output_dir, summary):
    summary_json_path = f"{output_dir}/summary.json"
    all_summaries = []
    if os.path.exists(summary_json_path):
        with open(summary_json_path) as summary_file:
            try:
                all_summaries = json.load(summary_file)
            except json.JSONDecodeError:
                all_summaries = []
    all_summaries.append(summary)
    logging.info(f"Writing summary to {summary_json_path}")
    with open(summary_json_path, mode="w") as summary_file:
        json.dump(all_summaries, summary_file)


def main():
    args = get_args()
    configuration = get_configuration(args.scenarios)
    output_dir = Path(args.output_dir).absolute()
    logging.info(f"Saving results to {output_dir}")
    output_dir.mkdir(exist_ok=True)
    for scenario, mode in configuration:
        logging.info(f"Running {scenario} {mode}")
        base_dir = Path(scenarios_config["scenarios"][scenario]["code_dir"])
        benchmark = scenarios_config["scenarios"][scenario]["benchmark"]
        command = scenarios_config["scenarios"][scenario]["command"]

        # logs are saved in the code/<model> dir
        logs_path = base_dir / "build" / "logs"

        # start timer
        total_time = 0
        start = time.time()
        if args.mode == "perf":
            # copy audit.config to get accuracy logs from performance mode
            # this is equivalent to running compliance TEST01
            shutil.copyfile("accuracy_from_perf.config", base_dir / "audit.config")

            accuracy_path = output_dir / "logs" / scenario / mode / "compliance" / "TEST01"
            # logs from performance are the same as accuracy in this mode
            performance_path = accuracy_path

            run_inference(base_dir, command, mode, False, scenario)
            evaluation_time = evaluate(base_dir)
            # move logs
            shutil.move(logs_path, accuracy_path)
            # remove audit
            os.remove(base_dir / "audit.config")
        else:
            # run accuracy
            logging.info("Running accuracy")
            run_inference(base_dir, command, mode, True, scenario)
            evaluation_time = evaluate(base_dir)
            accuracy_path = output_dir / "logs" / scenario / mode / "accuracy"
            shutil.move(logs_path, accuracy_path)
            if args.mode != "acc":
                logging.info("Running performance")
                run_inference(base_dir, command, mode, False, scenario)
                performance_path = (
                    output_dir / "logs" / scenario / mode / "performance" / "run_1"
                )
                shutil.move(logs_path, performance_path)

        # get summary
        precision = scenarios_config["scenarios"][scenario]["precision"]
        batch_size = scenarios_config["scenarios"][scenario]["batch_size"]
        total_time = time.time() - start
        results = get_results(accuracy_path, benchmark)
        units = units_map[mode]
        if args.mode != "acc":
            perf = get_performance(performance_path, mode)
            performance = perf['samples_per_seconds']
            results = verify_performance(perf['validity'], results)
        else:
            performance = None
        if "gptj" in scenario:
            thresholds = scenarios_config["benchmarks"]["gptj"]
            results["accuracy"] = (
                min(
                    results["rouge1"] / thresholds["rouge1"],
                    results["rouge2"] / thresholds["rouge2"],
                    results["rougeL"] / thresholds["rougeL"],
                )
                * 100
            )
        summary = {
            "model": benchmark,
            "scenario": scenario,
            "units": units,
            "performance": performance,
            "batch_size": batch_size,
            "precision": precision,
            "iterations": results["gen_num"],
            "dataset": scenarios_config["scenarios"][scenario]["dataset"],
            "total_time": total_time,
            "eval_time": evaluation_time,
            "warmup_time": 0,
            **results,
        }
        write_summary(output_dir, summary)
        shutil.rmtree(base_dir / "build")


if __name__ == "__main__":
    main()
