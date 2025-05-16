import re
import sys

result_file = sys.argv[1]

lines = open(result_file).readlines()


headers = lines[0::3]
results = lines[1::3]
def extract(header, result):
    return re.search(r"alignment_([\d]+)_([\d]+)\.fasta\.(.+)\.time", header), result.split()[0]


data = list(map(lambda p: extract(*p), zip(headers, results)))


data_dict = {}
for (para, time) in data:
    data_dict[(para.group(1), para.group(2), para.group(3))] = time

def time_str_to_seconds(time_str):
    parts = time_str.split(':')
    seconds_part = parts[-1]  # Always the last part
    
    # Handle seconds and milliseconds
    if '.' in seconds_part:
        seconds, milliseconds = map(float, seconds_part.split('.'))
        total = seconds + milliseconds / 1000
    else:
        total = float(seconds_part)
    
    # Add minutes if present
    if len(parts) >= 2:
        total += float(parts[-2]) * 60
    
    # Add hours if present
    if len(parts) >= 3:
        total += float(parts[-3]) * 3600
    
    return total

def create_latex_table(data_dict, columns = '100'):
    header = "\\begin{table}\n\\centering\n\\begin{tabular}{|c|c|c|c|}\n\\hline\n"
    header += "\\textbf{Tree Size} & \\textbf{IQ-Tree (s)} & \\textbf{PhyloGrad (s)} & \\textbf{Factor} \\\\\n\\hline\n"
    for size in [16,64,256,1024,4096]:
        iqtree_time = data_dict[(str(size), columns, 'iqtree')]
        phylograd_time = data_dict[(str(size), columns, 'phylograd')]
        
        iqtree_seconds = time_str_to_seconds(iqtree_time)
        phylograd_seconds = time_str_to_seconds(phylograd_time)
        factor = iqtree_seconds / phylograd_seconds
        
        header += f"{size} & {iqtree_seconds:.0f} & {phylograd_seconds:.0f} & {factor:.2f} \\\\\n\\hline\n"
    header += "\\end{tabular}\n\\caption{Benchmarking of IQ-TREE and PhyloGrad on "+columns+" columns}\n\\label{tab:benchmark_"+columns+"}\n\\end{table}"
    return header
print(create_latex_table(data_dict, columns = '500'))
print(create_latex_table(data_dict, columns = '100'))