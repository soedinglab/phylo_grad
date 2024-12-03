"""
    This script runs the benchmark and measures the time and memory usage of the program.
"""
import sys
import subprocess
import time

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
    
    # Get memory info
    if process.returncode != 0:
        print("ERROR")
        print(stderr.decode())
    print(f"Time: {elapsed_time:.2f}")
    print(f"Mem: {stdout.decode()}")

dtype = sys.argv[1]

assert dtype in ['f32', 'f64']

pytorch_backend = sys.argv[2]

files = sys.argv[3:]
assert(len(files) %2 == 0)

newick_files = files[0:len(files)//2]
fasta_files = files[len(files)//2:]

for fasta_file, newick_file in zip(fasta_files, newick_files):
    print("--Dataset--: ", fasta_file, newick_file)
    print("Rust:")
    run_and_measure(f'python optimize.py --fasta_amino {fasta_file} --newick {newick_file} --rust --{dtype}')
    print("Pytorch:")
    run_and_measure(f'python optimize.py --fasta_amino {fasta_file} --newick {newick_file} --{pytorch_backend} --{dtype}')
    print()