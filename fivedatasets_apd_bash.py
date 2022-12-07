"""
"""

import subprocess
import pdb
import sys
import copy
template = 'python fivedatasets_apd_run.py \
					--n_epochs 50\
					--n_classes 10\
					--n_tasks 5\
					--decay_rate 1\
					--mask_hyp 0. \
					--l1_hyp 4e-4 \
					--approx_hyp 100.\
					--clustering_iter 5\
					--k_centroides 2\
					--e_gap_hyp 1e-2 \
					--results_path results/fivedatasets'

processes = []
arg_container = []
command = template.format(*arg_container)
process = subprocess.call(command, shell=True)
#process = subprocess.Popen(command, shell=True)
#output = process.wait()
