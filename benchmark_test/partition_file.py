import sys

L = int(sys.argv[1])
format = sys.argv[2]
output_file = sys.argv[3]

if format == "nexus":
    model = "GTR20+FO"
    with open(output_file, 'w') as f:
        f.write("#nexus\n")
        f.write("begin sets;\n")
        for i in range(1,L+1):
            f.write(f"    charset mypart{i} = {i}-{i};\n")
        f.write("charpartition mine = ")
        for i in range(1,L):
            f.write(f"{model}:mypart{i}, ")
        f.write(f"{model}:mypart{L};\n")
        f.write("end;\n")
else:

    model = "GTR20+FO" if format == "iqtree" else "PROTGTR+FO"

    with open(output_file, 'w') as f:
        for i in range(1,L+1):
            f.write(f"{model}, col{i} = {i}-{i}\n")