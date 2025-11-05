import re
import sys

input_file = sys.argv[1]
with open(input_file, 'r') as f:
    content = f.read()

pattern = r"GradientTime=(\d+)"
matches = re.findall(pattern, content)

total_runtime = re.search(r"Total wall-clock time used: ([\d\.]+) sec", content)
if total_runtime is None: # raxml log
    total_runtime = re.search(r"Elapsed time: ([\d\.]+) seconds", content)
best_score = re.search(r"BEST SCORE FOUND : ([\d\.\-]+)", content)
if best_score is None: # raxml log
    best_score = re.search(r"Final LogLikelihood: ([\d\.\-]+)", content)

print("Number of gradient calculations:", len(matches))
print("Average gradient time (ms):", sum(map(float, matches)) / len(matches) / 1000.0)
print("Total gradient time (s):", sum(map(float, matches)) / 1_000_000.0)
print("Total wall-clock time used (s):", total_runtime.group(1) if total_runtime else "N/A")
print("Best log likelihood found:", best_score.group(1) if best_score else "N/A")