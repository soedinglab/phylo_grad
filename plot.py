import re
import sys
import matplotlib.pyplot as plt
import numpy as np

lines = open(sys.argv[1]).readlines()

output_time = sys.argv[2]
output_mem = sys.argv[3]
output_ratio = sys.argv[4]


datasets = lines[0::10]
rust_time = lines[2::10]
rust_mem = lines[3::10]
python_time = lines[6::10]
python_mem = lines[7::10]

def extract_number(iter):
    return np.array([float(re.search(r'\d+\.?\d+', x).group(0)) for x in iter])

def extract_tree_size(iter):
    return np.array([int(re.search(r'tree_(\d+)', x).group(1)) for x in iter])

ax = plt.gca()
ax.scatter(extract_tree_size(datasets), extract_number(rust_time), label='PhyloGrad')
ax.scatter(extract_tree_size(datasets), extract_number(python_time), label='Pytorch')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Tree size')
ax.set_ylabel('Runtime (s)')
ax.legend()
plt.savefig(output_time)
ax.clear()


ax = plt.gca()
ax.scatter(extract_tree_size(datasets), extract_number(python_time) / extract_number(rust_time))
ax.set_xscale('log')
ax.set_xlabel('Tree size')
ax.set_ylabel('Runtime Pytorch (s) / Runtime PhyloGrad (s)')
plt.savefig(output_ratio)
ax.clear()

ax = plt.gca()
ax.scatter(extract_tree_size(datasets), extract_number(rust_mem) / 1024**2, label='PhyloGrad')
ax.scatter(extract_tree_size(datasets), extract_number(python_mem) / 1024**2, label='Pytorch')
ax.set_xscale('log')
ax.set_xlabel('Tree size')
ax.set_ylabel('Peak memory (GiB)')
ax.legend()
plt.savefig(output_mem)

