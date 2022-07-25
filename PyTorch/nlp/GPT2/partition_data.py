# Copyright (c) 2022, Habana Labs Ltd.  All rights reserved.
import sys
import os
import random
import shutil

length = len(sys.argv)
if length > 4:
    print("Incorrect number of parameters")
    exit(0)

input_folder = str(sys.argv[1])
output_folder = str(sys.argv[2])
data_percent = float(sys.argv[3])

for root, dire, files in os.walk(input_folder):
    if not dire:
        random_files = random.sample(files, int(len(files)*data_percent))
        for name in random_files:
            shutil.move(os.path.join(root,name),output_folder)
