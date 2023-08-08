#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###############################################################################

from huggingface_hub import snapshot_download
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Pretrained weight downloader for Huggingface models')
parser.add_argument('--model', '-m', type=str, help='Target model in repo/model format', required=True)
parser.add_argument('--weights', type=str, help="Weight dir for all pretrained models", required=True)
parser.add_argument('--files', action='append', help="Download only files matching pattern", default=[])
args = parser.parse_args()
if len(args.files) == 0:
    args.files = ['*.json', '*.bin', '*.txt']
snapshot_download(repo_id=args.model, cache_dir=args.weights, resume_download=True, local_files_only=False, allow_patterns=args.files)
