#! /usr/bin/env python

import numpy as np
import os

judge_path = "Data/judge_data"
judge_dataset = np.empty((0,1))
judge_key = '.txt'
cnt = 0

for j_dir_name, j_sub_dirs, j_files in os.walk(judge_path): 
    for jf in j_files:
        if judge_key == jf[-len(judge_key):]:
            f = open(os.path.join(j_dir_name, jf), 'r')
            n = f.read()
            judge_dataset = np.append(judge_dataset, n)
            cnt += 1
print(judge_dataset)

