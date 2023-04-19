import json
import sys

results = json.load(open("./latest/results.json"))
print(results[sys.argv[1]])
