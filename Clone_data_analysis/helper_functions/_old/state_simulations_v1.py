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
            
        # Convert dictionary to list in correct order
        probs = [initial_state_probs.get(state, 0.0) for state in [
            'state_00', 'state_01', 'state_10', 'state_11'
        ]]
        
        # Check if we're dealing with PyMC variables
        self.using_pymc = any(hasattr(p, 'type') for p in probs)
        
        if not self.using_pymc:
            if not np.isclose(sum(probs), 1.0):
                raise ValueError("Initial probabilities must sum to 1")
        
        self.params = params
        self.initial_probs = probs
        
    def _transition_cell(self, state: Tuple[int, int]) -> Tuple[int, int]:
        """Apply state transition to a single cell"""
        r = np.random.random()
        
        if state == (0,0):
            if r < self.params.q00_01:
                return (0,1)
            elif r < self.params.q00_01 + self.params.q00_10:
                return (1,0)
        elif state == (0,1):
            if r < self.params.q01_00:
                return (0,0)
            elif r < self.params.q01_00 + self.params.q01_11:
                return (1,1)
        elif state == (1,0):
            if r < self.params.q10_00:
                return (0,0)
            elif r < self.params.q10_00 + self.params.q10_11:
                return (1,1)
        elif state == (1,1):
            if r < self.params.q11_01:
                return (0,1)
            elif r < self.params.q11_01 + self.params.q11_10:
                return (1,0)
                
        return state
        
    def simulate_clone(self, n_divisions: int) -> Tuple[float, float]:
        """Simulate a single clone for n divisions and return (f1, f2)"""
        if self.using_pymc:
            raise ValueError("Direct simulation not supported with PyMC variables. Use probabilistic calculations instead.")
            
        # Initialize first cell
        states = [(np.random.choice(4, p=self.initial_probs) // 2,
                  np.random.choice(4, p=self.initial_probs) % 2)]
        
        # Simulate divisions
        for _ in range(n_divisions):
            new_states = []
            for state in states:
                # Each cell divides into two
                for _ in range(2):
                    new_state = self._transition_cell(state)
                    new_states.append(new_state)
            states = new_states
            
        # Calculate marginal frequencies
        f1 = sum(state[0] for state in states) / len(states)
        f2 = sum(state[1] for state in states) / len(states)
        
        return f1, f2

def run_simulations(params: TransitionParams, initial_probs: List[float],
                   n_sims: int, divisions: List[int]) -> dict:
    """Run multiple simulations and collect results"""
    sim = CellSimulation(params, initial_probs)
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