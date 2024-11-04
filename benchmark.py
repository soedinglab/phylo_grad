import sys
import subprocess
import psutil
import time

def run_and_monitor(command):
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
    
    print(f"Time: {elapsed_time:.2f}")
    print(f"Mem: {stdout.decode()}")
        
files = sys.argv[1:]
assert(len(files) %2 == 0)

newick_files = files[0:len(files)//2]
fasta_files = files[len(files)//2:]

for fasta_file, newick_file in zip(fasta_files, newick_files):
    print("--Dataset--: ", fasta_file, newick_file)
    print("Rust:")
    run_and_monitor(f'python optimize.py --fasta_amino {fasta_file} --newick {newick_file} --rust --f64')
    print("Pytorch:")
    run_and_monitor(f'python optimize.py --fasta_amino {fasta_file} --newick {newick_file} --pytorch --f64')
    print()