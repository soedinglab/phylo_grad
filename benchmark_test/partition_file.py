import sys

L = int(sys.argv[1])
format = sys.argv[2]
output_file = sys.argv[3]

model = "GTR20+FO" if format == "iqtree" else "PROTGTR+FO"

with open(output_file, 'w') as f:
    for i in range(1,L+1):
        f.write(f"{model}, col{i} = {i}-{i}\n")