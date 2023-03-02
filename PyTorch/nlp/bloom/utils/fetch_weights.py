#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###############################################################################

from huggingface_hub import snapshot_download
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Pretrained weight downloader for Huggingface models')
parser.add_argument('--model', '-m', type=str, choices=['bigscience/' + m for m in['bloom-560m', 'bloom-1b7', 'bloom-3b', 'bloom-7b1', 'bloom']], help='Model', required=True)
parser.add_argument('--weights', type=str, help="Weight dir for all pretrained models", required=True)
args = parser.parse_args()
snapshot_download(repo_id=args.model, cache_dir=args.weights, resume_download=True, local_files_only=False, allow_patterns=["*.json", "*.bin"])
