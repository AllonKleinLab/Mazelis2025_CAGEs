import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import seaborn as sns

@dataclass
class TransitionParams:
    q0_1: float  # 0 -> 1
    q1_0: float  # 1 -> 0
    
    def validate(self):
        """Check transition probabilities are valid"""
        # Check if any of the parameters is a PyTensor variable
        if any(hasattr(getattr(self, attr), 'type') 
               for attr in ['q0_1', 'q1_0']):
            return True
            
        # For regular float values, perform normal validation
        return self.q0_1 <= 1 and self.q1_0 <= 1

class CellSimulation:
    def __init__(self, params: TransitionParams, initial_state_probs: dict):
        if not params.validate():
            raise ValueError("Invalid transition probabilities")
        
        self.params = params
        self._init_probs = [
            initial_state_probs['state_0'],
            initial_state_probs['state_1']
        ]
        
        # Pre-compute transition thresholds
        self._thresholds = [
            params.q0_1,  # state 0
            params.q1_0   # state 1
        ]
        
        # Results lookup table: [state][threshold_case] -> new_state
        self._results = [
            [1, 0],  # from 0: 1, 0
            [0, 1]   # from 1: 0, 1
        ]
    
    def _transition_cell(self, state):
        """Ultra-fast state transition using lookup tables"""
        r = np.random.random()
        if r < self._thresholds[state]:
            return self._results[state][0]
        return self._results[state][1]
        
    def simulate_clone(self, n_divisions: int):
        """Fast simulation using integer states"""
        if n_divisions == 0:
            state = np.random.choice(2, p=self._init_probs)
            return state
        
        states = [np.random.choice(2, p=self._init_probs)]
        
        for _ in range(n_divisions):
            new_states = []
            for state in states:
                new_states.append(self._transition_cell(state))
                new_states.append(self._transition_cell(state))
            states = new_states
        
        return sum(states) / len(states)  # Return fraction of 1s

def run_simulations(params: TransitionParams, initial_state_probs: dict,
                   n_sims: int, divisions: List[int]) -> dict:
    """Run multiple simulations and collect results"""
    sim = CellSimulation(params, initial_state_probs)
    results = {d: [] for d in divisions}
    
    for _ in range(n_sims):
        for d in divisions:
            f = sim.simulate_clone(d)
            results[d].append(f)
            
    return results

def plot_binary_threshold(results: dict, thresh: float = 0.5):
    """Plot binary threshold dynamics for two categories based on fraction threshold"""
    fig, ax = plt.subplots(figsize=(4, 4))
    divisions = sorted(results.keys())
    
    # Calculate fractions for each category over time
    hi = []
    lo = []
    
    for div in divisions:
        points = np.array(results[div])
        hi.append(np.mean(points > thresh))
        lo.append(np.mean(points <= thresh))
    
    ax.plot(divisions, hi, 'r-o', label=f'f-hi')
    ax.plot(divisions, lo, 'b-o', label=f'f-lo')
    
    ax.set_xlabel('Divisions')
    ax.set_ylabel('Fraction of clones')
    ax.set_title('Binary threshold dynamics')
    ax.legend(loc='upper left')
    
    return fig

def visualize_results(results: dict):
    """Create histogram and distribution visualizations"""
    n_times = len(results)
    fig, axes = plt.subplots(1, n_times, figsize=(4*n_times, 4))
    
    for i, (div, points) in enumerate(sorted(results.items())):
        points = np.array(points)
        
        # Distribution plot
        sns.histplot(data=points, ax=axes[i], bins=20)
        axes[i].set_title(f'After {div} divisions')
        axes[i].set_xlabel('Fraction of state 1')
        axes[i].set_ylabel('Count')
    
    plt.tight_layout()
    return fig


# # Example usage
# params = TransitionParams(
#     q0_1=0.2,
#     q1_0=0.1
# )
#
# initial_state_probs = {
#     'state_0': 0.5,
#     'state_1': 0.5
# }

# divisions = [0, 2, 4, 6]  # Division numbers to sample
# n_sims = 1000

# results = run_simulations(params, initial_state_probs, n_sims, divisions)
# fig1 = visualize_results(results)
# fig2 = plot_binary_threshold(results, thresh=0.5)
# plt.show()