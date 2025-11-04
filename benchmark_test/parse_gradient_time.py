import re
import sys

input_file = sys.argv[1]
with open(input_file, 'r') as f:
    content = f.read()

pattern = r"GradientTime=(\d+)"
matches = re.findall(pattern, content)

total_runtime = re.search(r"Total wall-clock time used: ([\d\.]+) sec", content)

print("Number of gradient calculations:", len(matches))
print("Average gradient time (ms):", sum(map(float, matches)) / len(matches) / 1000.0)
print("Total gradient time (s):", sum(map(float, matches)) / 1_000_000.0)
print("Total wall-clock time used (s):", total_runtime.group(1))
