import json

with open('results/simulation_results.json', 'r') as f:
    data = json.load(f)

for algo, metrics in data['results'].items():
    print(f"Algorithm: {algo}")
    print(metrics['stats'].keys())
    for k, v in metrics['stats'].items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v}")
