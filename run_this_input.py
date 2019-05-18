#!/usr/bin/env python3
import os
import sys
import re

fname = 'input_file.prm'

f = open(fname)

regexp = re.compile(r'set dimension.*?([1-3.-]+)')

dim = -1
with open(fname) as f:
    for line in f:
        match = regexp.match(line)
        if match:
            dim = int(match.group(1))
            break
if dim == -1:
    sys.exit("No valid 'set dimension = [1-3]' line found")



bin_path = './bin/PHiLiP_'+str(dim)+'D'

command = bin_path + ' -p ' + fname
print("Running: " + command)
os.system(command)
