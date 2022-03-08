from network_functions import *
from States import states, forbidden_transition_attempts, allowed_transition_attempts
import json

transition_matrices = {}

solver, shape = 'human', 'SPT'
for size in ['Medium', 'Large']:
    if os.path.exists(size + '_transitions.txt'):
        with open(size + '_transitions.txt', 'r') as f:
            paths = json.load(f)

    state_order = sorted(states + forbidden_transition_attempts + allowed_transition_attempts)
    my_network = Network(state_order, name='_'.join(['network', solver, size, shape]))
    my_network.add_paths(paths)
    transition_matrices[size] = my_network.transition_matrix().toarray()
    my_network.plot_transition_matrix(title=size)
    # TODO: in the transition matrix plot (in line 18), get rid of unoccupied states
