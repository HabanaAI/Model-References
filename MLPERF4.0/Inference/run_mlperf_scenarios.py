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
compliance_tests_map = {"llama2-70b-99.9": ["TEST06"]}


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
    parser.add_argument("--mode", type=str, choices=["full", "perf", "acc"],
                        default="full", help="dev options to shorten test time")
    parser.add_argument("--compliance", action="store_true")

    args = parser.parse_args()
    return args


def run_inference(base_dir, command, mode, accuracy, scenario, std_out_logs):
    command += f" --scenario {mode}"
    if accuracy:
        command += " --accuracy"
    command += f" 2>&1 | tee {std_out_logs}"
    logging.info(command)
    try:
        subprocess.run(command, check=True, shell=True, cwd=base_dir)
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed running {scenario}_{mode}")
        sys.exit(1)


def evaluate(base_dir):
    start_time = time.time()
    # Assuming script naming convention is consistent between models
    command = "python evaluation.py | tee -a ./build/logs/accuracy.txt"
    logging.info(command)
    try:
        subprocess.run(command, check=True, shell=True, cwd=base_dir)
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed evaluating {base_dir}")
        sys.exit(1)
    return time.time() - start_time


def verify_thresholds(benchmark, results: typing.Dict[str, typing.Any]):
    error = ""
    valid = True
    thresholds = scenarios_config["benchmarks"][benchmark]
    for metric, threshold in thresholds.items():
        if metric in results and results[metric] < threshold:
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


def get_performance(performance_path, mode, code_dir):
    perf = {}
    text = open(performance_path / "mlperf_log_summary.txt").read()
    if mode == "Offline":
        if code_dir == "llama":
            perf_pattern = "Tokens per second: (.+?)\n"
        else:
            perf_pattern = "Samples per second: (.+?)\n"
    else:
        if code_dir == "llama":
            perf_pattern = "Completed tokens per second                 : (.+?)\n"
        else:
            perf_pattern = "Scheduled samples per second : (.+?)\n"

    validity_pattern = "Result is : (.+?)\n"
    perf['result'] = re.search(perf_pattern, text).group(1)
    perf['validity'] = re.search(validity_pattern, text).group(1)

    return perf


def verify_performance(perf_validity, results: typing.Dict[str, typing.Any]):
    if perf_validity == "INVALID":
        results["valid"] = False
        results["error"] = "invalid"
    return results


def get_warmup_time(output_file):
    try:
        with open(output_file, 'r') as file:
            log_content = file.read()
            match = re.search(r'Warmup took (\d+(\.\d+)?)', log_content)
            warmup_time = float(match.group(1)) if match else 0

            return warmup_time
    except FileNotFoundError:
        return 0


def get_estimated_performance(output_file):
    try:
        with open(output_file, 'r') as file:
            log_content = file.read()
            match = re.search(
                r'Estimated performance for accuracy run is (\d+(\.\d+)?)', log_content)
            estimated_performance = float(match.group(1)) if match else 0
            return estimated_performance
    except FileNotFoundError:
        return 0


def get_gen_num_for_perf(output_file):
    try:
        with open(output_file, 'r') as file:
            log_content = file.read()
            match = re.search(
                r'performance_sample_count : (\d+(\.\d+)?)', log_content)
            gen_num = float(match.group(1)) if match else 0
            return gen_num
    except FileNotFoundError:
        return 0


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


def run_compliance_test(test, base_dir, output_dir, logs_path, command, mode, scenario, mlperf_path, std_out_logs):
    logging.info(f"Running compliance {test}")
    source_path = os.path.join(
        mlperf_path, f"compliance/nvidia/{test}/audit.config")
    destination_path = os.path.join(base_dir, "audit.conf")
    shutil.copyfile(source_path, destination_path)
    run_inference(base_dir, command, mode, False, scenario, std_out_logs)
    evaluate(base_dir)
    compliance_path = (
        output_dir / "logs" / scenario / mode / "compliance" / test
    )
    shutil.move(str(logs_path), str(compliance_path))
    # remove audit
    os.remove(base_dir / "audit.conf")


def run_subprocess(cmd, cwd):
    if cmd is not None:
        try:
            subprocess.run(cmd, check=True, shell=True, cwd=cwd)
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed running {cmd}")
            sys.exit(1)
    return False


def main():
    args = get_args()
    configuration = get_configuration(args.scenarios)
    output_dir = Path(args.output_dir).absolute()
    logging.info(f"Saving results to {output_dir}")
    output_dir.mkdir(exist_ok=True)
    init_setup_done = False
    for scenario, mode in configuration:
        code_dir = scenarios_config["scenarios"][scenario]["code_dir"]
        base_dir = Path(code_dir)
        # Initialize setup
        if init_setup_done is False:
            init_setup_cmd = scenarios_config["scenarios"][scenario].get(
                "init_setup", None)
            init_setup_done = run_subprocess(init_setup_cmd, base_dir)
        # Initialize scenario
        init_scenario = scenarios_config["scenarios"][scenario].get(
            f"init_{mode}", None)
        init_scenario_cmd = None if init_scenario is None else f"{init_scenario} {output_dir}"
        run_subprocess(init_scenario_cmd, base_dir)

        logging.info(f"Running {scenario} {mode}")
        benchmark = scenarios_config["scenarios"][scenario]["benchmark"]
        command = scenarios_config["scenarios"][scenario]["command"]
        std_out_logs = output_dir / "std_out_logs.txt"
        # logs are saved in the code/<model> dir
        logs_path = base_dir / "build" / "logs"

        # start timer
        total_time = 0
        start = time.time()
        results = {}
        if args.mode != "perf":
            logging.info("Running accuracy")
            run_inference(base_dir, command, mode,
                          True, scenario, std_out_logs)
            evaluation_time = evaluate(base_dir)
            accuracy_path = output_dir / "logs" / scenario / mode / "accuracy"
            shutil.move(str(logs_path), str(accuracy_path))

            results = get_results(accuracy_path, benchmark)
            if args.mode != "full":
                performance = get_estimated_performance(std_out_logs)

        if args.mode != "acc":
            logging.info("Running performance")
            run_inference(base_dir, command, mode,
                          False, scenario, std_out_logs)
            performance_path = (
                output_dir / "logs" / scenario / mode / "performance" / "run_1"
            )
            shutil.move(str(logs_path), str(performance_path))

            perf = get_performance(performance_path, mode, code_dir)
            performance = perf['result']
            results = verify_performance(perf['validity'], results)
            if args.mode != "full":
                results["gen_num"] = get_gen_num_for_perf(std_out_logs)
                evaluation_time = 0

        total_time = time.time() - start

        if args.compliance and benchmark in compliance_tests_map:
            for test in compliance_tests_map[benchmark]:
                run_compliance_test(test, base_dir, output_dir, logs_path,
                                    command, mode, scenario, args.mlperf_path, std_out_logs)

        # get summary
        precision = scenarios_config["scenarios"][scenario]["precision"]
        batch_size = scenarios_config["scenarios"][scenario].get(
                f"batch_size_{mode}", None)
        if batch_size is None:
            batch_size = scenarios_config["scenarios"][scenario]["batch_size"]
        units = "Tokens/s" if code_dir == "llama" else units_map[mode]

        if "sd-xl" in scenario and args.mode != "perf" and results["error"] == "":
            thresholds = scenarios_config["benchmarks"][benchmark]
            error = ""
            valid = False
            if results["FID_SCORE"] > thresholds["FID_SCORE_MAX"]:
                error += "FID_SCORE_MAX"
            if results["CLIP_SCORE"] > thresholds["CLIP_SCORE_MAX"]:
                error += "CLIP_SCORE_MAX"
            if error != "":
                results["error"] = error
                results["valid"] = valid

        warmup_time = get_warmup_time(std_out_logs)
        summary = {
            "model": benchmark,
            "scenario": scenario,
            "units": units,
            "performance": performance,
            "batch_size": batch_size,
            "precision": precision,
            "iterations": results["gen_num"],
            "dataset": scenarios_config["scenarios"][scenario]["dataset"],
            "total_time": round(total_time, 2),
            "eval_time": round(evaluation_time, 2),
            "warmup_time": round(warmup_time, 2),
            **results,
        }
        write_summary(output_dir, summary)
        shutil.rmtree(base_dir / "build")


if __name__ == "__main__":
    main()
