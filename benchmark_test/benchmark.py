"""
    This script runs the benchmark and measures the time and memory usage of the program.
"""
import sys
import subprocess
import time
import re
import json

def run_and_measure(command):
    # Start time
    start_time = time.time()

    # Start the process
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    
    stdout, stderr = process.communicate()
    
    # End time
    end_time = time.time()
    
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    
    if process.returncode != 0:
        print("Error running command:" + command)
        print("stderr:")
        print(stderr.decode())
        print("stdout:")
        print(stdout.decode())
        return (0.0, 0.0)
    else:
        return (float(elapsed_time), float(stdout.decode))
    
dtypes = sys.argv[1] # comma separated list of dtypes (no spaces)
dtypes = dtypes.split(',')
assert all([dtype in ['f32', 'f64'] for dtype in dtypes])

backends = sys.argv[2] # comma separated list of backends (no spaces)
backends = backends.split(',')
assert all([backend in ['pytorch', 'rust', 'pytorch_gpu'] for backend in backends])

files = sys.argv[3:]
assert(len(files) %2 == 0)

newick_files = files[0:len(files)//2]
fasta_files = files[len(files)//2:]

measurements = {}

for fasta_file, newick_file in zip(fasta_files, newick_files):
    match = re.search(r'(\d+)_(\d+)', fasta_file)
    if match:
        L = int(match.group(2))
        N = int(match.group(1))
    else:
        raise ValueError("Could not parse L and N from fasta file")
    
    for dtype in dtypes:
        for backend in backends:
            command = f"python3 benchmark.py --{backend} --{dtype} --newick {newick_file} --amino_fasta {fasta_file}"
            t, mem = run_and_measure(command)
            measurements[(backend, dtype, L, N)] = (t, mem)

print(json.dumps(measurements))