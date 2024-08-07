###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

import json

# Opening JSON file
import numpy as np
# returns JSON object as
# a dictionary

for i in range(160):
    train_dict = {}
    train_dict["input_ids"]=np.random.randint(8192, size=(8192)).tolist()
    train_dict["labels"]=np.random.randint(8192, size=(8192)).tolist()
    tweets = []
    with open("./train_warmup.json", "a") as outfile:
        json.dump(train_dict, outfile)
        outfile.write('\n')

for i in range(173):
    train_dict = {}
    train_dict["input_ids"]=np.random.randint(8192, size=(8192)).tolist()
    train_dict["labels"]=[-100] * 8192
    tweets = []
    with open("./eval_warmup.json", "a") as outfile:
        json.dump(train_dict, outfile)
        outfile.write('\n')

# Closing file
