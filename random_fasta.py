import sys
import Bio.Phylo as Phylo
import input
import random

newick_file = sys.argv[1]
L = (int)(sys.argv[2])
output_file = sys.argv[3]

random.seed(42)

amino_acids = list(input.amino_mapping.keys())

newick = Phylo.read(newick_file, 'newick')

with open(output_file, 'w') as f:
    for terminal in newick.get_terminals():
        f.write('>' + terminal.name + '\n')
        seq = ''.join([random.choice(amino_acids) for _ in range(L)])
        f.write(seq + '\n')
        