"""
    This script runs the benchmark and measures the time and memory usage of the program.
"""
import sys
import subprocess
import time
import re
import pickle

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
        print("Error running command:" + command, file=sys.stderr)
        print("stderr:", file=sys.stderr)
        print(stderr.decode(), file=sys.stderr)
        print("stdout:", file=sys.stderr)
        print(stdout.decode(), file=sys.stderr)
        return (0.0, 0.0)
    else:
        return (float(elapsed_time), float(stdout.decode()))
    
output = sys.argv[1]

dtypes = sys.argv[2] # comma separated list of dtypes (no spaces)
dtypes = dtypes.split(',')
assert all([dtype in ['f32', 'f64'] for dtype in dtypes])

backends = sys.argv[3] # comma separated list of backends (no spaces)
backends = backends.split(',')
assert all([backend in ['pytorch', 'rust', 'pytorch_gpu', 'jax_gpu'] for backend in backends])

files = sys.argv[4:]
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
            command = f"python3 optimize.py --{backend} --{dtype} --newick {newick_file} --fasta_amino {fasta_file}"
            t, mem = run_and_measure(command)
            measurements[(backend, dtype, L, N)] = (t, mem)

pickle.dump(measurements, open(output, 'wb'))