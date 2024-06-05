import os
import time
import json
import argparse

def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlperf-accuracy-file", default="build/logs/results.json", help="path to results.json")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    with open(args.mlperf_accuracy_file, "r") as file:
        data = json.load(file)
    
    acc_results = data.get("accuracy_results", {"CLIP_SCORE": 0.0, "FID_SCORE": 0.0})
    args = data.get("args", {})
        
    acc_results["gen_num"] = args["count"]
    print("\nResults\n")
    print(acc_results)
if __name__ == "__main__":
    main()
