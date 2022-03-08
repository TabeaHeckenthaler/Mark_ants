import json
import os

for size in ['Medium', 'Large']:
    if os.path.exists(size + '_transitions.txt'):
        with open(size + '_transitions.txt', 'r') as f:
            states_series = json.load(f)

        print(states_series)
