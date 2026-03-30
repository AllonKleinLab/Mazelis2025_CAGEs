import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import seaborn as sns

@dataclass
class TransitionParams:
    q00_01: float  # (0,0) -> (0,1)
    q00_10: float  # (0,0) -> (1,0)
    q01_00: float  # (0,1) -> (0,0)
    q01_11: float  # (0,1) -> (1,1)
    q10_00: float  # (1,0) -> (0,0)
    q10_11: float  # (1,0) -> (1,1)
    q11_01: float  # (1,1) -> (0,1)
    q11_10: float  # (1,1) -> (1,0)
    
    def validate(self):
        """Check transition probabilities are valid"""
        # Check if any of the parameters is a PyTensor variable
        if any(hasattr(getattr(self, attr), 'type') 
               for attr in ['q00_01', 'q00_10', 'q01_00', 'q01_11',
                          'q10_00', 'q10_11', 'q11_01', 'q11_10']):
            return True
            
        # For regular float values, perform normal validation
        conditions = [
            self.q00_01 + self.q00_10 <= 1,
            self.q01_00 + self.q01_11 <= 1,
            self.q10_00 + self.q10_11 <= 1,
            self.q11_01 + self.q11_10 <= 1
        ]
        return all(conditions)

class CellSimulation:
    def __init__(self, params: TransitionParams, initial_state_probs: dict):
        if not params.validate():
            raise ValueError("Invalid transition probabilities")
        
        self.params = params
        self._init_probs = [
            initial_state_probs['state_00'],
            initial_state_probs['state_01'],
            initial_state_probs['state_10'],
            initial_state_probs['state_11']
        ]
        
        # Pre-compute transition thresholds and results
        self._thresholds = [
            (params.q00_01, params.q00_01 + params.q00_10),  # state 0 (00)
            (params.q01_00, params.q01_00 + params.q01_11),  # state 1 (01)
            (params.q10_00, params.q10_00 + params.q10_11),  # state 2 (10)
            (params.q11_01, params.q11_01 + params.q11_10)   # state 3 (11)
        ]
        
        # Results lookup table: [state][threshold_case] -> new_state
        self._results = [
            [1, 2, 0],  # from 00: 01, 10, 00
            [0, 3, 1],  # from 01: 00, 11, 01
            [0, 3, 2],  # from 10: 00, 11, 10
            [1, 2, 3]   # from 11: 01, 10, 11
        ]
    
    def _transition_cell(self, state):
        """Ultra-fast state transition using lookup tables"""
        r = np.random.random()
        t1, t2 = self._thresholds[state]
        if r < t1: return self._results[state][0]
        if r < t2: return self._results[state][1]
        return self._results[state][2]
        
    def simulate_clone(self, n_divisions: int):
        """Fast simulation using integer states"""
        if n_divisions == 0:
            state = np.random.choice(4, p=self._init_probs)
            return state // 2, state % 2
        
        states = [np.random.choice(4, p=self._init_probs)]
        
        for _ in range(n_divisions):
            new_states = []
            for state in states:
                new_states.append(self._transition_cell(state))
                new_states.append(self._transition_cell(state))
            states = new_states
        
        # Convert back to binary representation for averaging
        return (sum(s // 2 for s in states) / len(states), 
                sum(s % 2 for s in states) / len(states))

def run_simulations(params: TransitionParams, initial_state_probs: dict,
                   n_sims: int, divisions: List[int]) -> dict:
    """Run multiple simulations and collect results"""
    sim = CellSimulation(params, initial_state_probs)
    results = {d: [] for d in divisions}
    
    for _ in range(n_sims):
        for d in divisions:
            f1, f2 = sim.simulate_clone(d)
            results[d].append((f1, f2))
            
    return results

def plot_binary_threshold(results: dict, thresh1: float = 0.5, thresh2: float = 0.5):
    """Plot binary threshold dynamics for four categories based on f1, f2 thresholds"""
    fig, ax = plt.subplots(figsize=(4, 4))
    divisions = sorted(results.keys())
    
    # Calculate fractions for each category over time
    hi_hi = []
    hi_lo = []
    lo_hi = []
    lo_lo = []
    
    for div in divisions:
        points = np.array(results[div])
        hi_hi.append(np.mean((points[:,0] > thresh1) & (points[:,1] > thresh2)))
        hi_lo.append(np.mean((points[:,0] > thresh1) & (points[:,1] <= thresh2)))
        lo_hi.append(np.mean((points[:,0] <= thresh1) & (points[:,1] > thresh2)))
        lo_lo.append(np.mean((points[:,0] <= thresh1) & (points[:,1] <= thresh2)))
    
    ax.plot(divisions, hi_hi, 'r-o', label=f'f1-hi, f2-hi')
    ax.plot(divisions, hi_lo, 'g-o', label=f'f1-hi, f2-lo')
    ax.plot(divisions, lo_hi, 'b-o', label=f'f1-lo, f2-hi')
    ax.plot(divisions, lo_lo, 'k-o', label=f'f1-lo, f2-lo')
    
    ax.set_xlabel('Divisions')
    ax.set_ylabel('Fraction of clones')
    ax.set_title('Binary threshold dynamics')
    ax.legend(loc='upper left')
    
    return fig

def visualize_results(results: dict):
    """Create heatmap and marginal distribution visualizations"""
    n_times = len(results)
    fig, axes = plt.subplots(2, n_times, figsize=(4*n_times, 8))
    
    # Plot heatmaps
    for i, (div, points) in enumerate(sorted(results.items())):
        points = np.array(points)
        
        # 2D histogram
        h, xedges, yedges = np.histogram2d(points[:,0], points[:,1], 
                                         bins=20, range=[[0,1], [0,1]])
        
        sns.heatmap(h.T, ax=axes[0,i], cmap='viridis')
        axes[0,i].set_title(f'After {div} divisions')
        axes[0,i].set_xlabel('f1')
        axes[0,i].set_ylabel('f2')
        
        # Marginal distributions
        for j, f in enumerate([points[:,0], points[:,1]]):
            sns.kdeplot(data=f, ax=axes[1,i])
            axes[1,i].set_title(f'Marginal distributions\nafter {div} divisions')
            axes[1,i].set_xlabel('Frequency')
            axes[1,i].set_ylabel('Density')
    
    plt.tight_layout()
    return fig

# # Example usage
# params = TransitionParams(
#     q00_01=0.2, q00_10=0.2,
#     q01_00=0.1, q01_11=0.2,
#     q10_00=0.1, q10_11=0.2,
#     q11_01=0.1, q11_10=0.1
# )

# initial_state_probs = {
#     'state_00': 0.25,
#     'state_01': 0.25,
#     'state_10': 0.25,
#     'state_11': 0.25
# }  # Equal probability for all states
# divisions = [0, 2, 4, 6]  # Division numbers to sample
# n_sims = 1000

# results = run_simulations(params, initial_state_probs, n_sims, divisions)
# fig1 = visualize_results(results)
# fig2 = plot_binary_threshold(results, thresh1=0.5, thresh2=0.5)
# plt.show()