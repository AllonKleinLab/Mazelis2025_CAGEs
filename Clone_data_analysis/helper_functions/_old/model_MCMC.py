# Import our previous simulation code
from state_simulations_v1 import TransitionParams, run_simulations

import pymc as pm
import numpy as np
from typing import List, Dict
import arviz as az

class CellModelInference:
    def __init__(self, data_df, threshold=0.05):
        """
        Initialize model inference with experimental data.
        
        Args:
            data_df: DataFrame with columns:
                - timepoint: division number
                - f1_hi: count of f1>thresh cells
                - f2_hi: count of f2>thresh cells  
                - f1f2_hi: count of both high
                - f1f2_lo: count of both low
            threshold: Threshold value to determine high/low states (default 0.5)
        """
        self.data_df = data_df
        self.threshold = threshold
        self.observed_counts = self._process_dataframe()
        print(f"Initialized with data containing {len(data_df)} timepoints")
        print(f"Using threshold value of {threshold}")
        
    def _process_dataframe(self):
        """
        Convert DataFrame to dictionary of count arrays.
        Each array contains [hihi, hilo, lohi, lolo] counts for a timepoint.
        """
        counts_dict = {}
        for timepoint, row in self.data_df.iterrows():
            # Calculate counts for each state
            hihi = row['f1f2_hi']  # Both factors high
            hilo = row['f1_hi'] - row['f1f2_hi']  # Only f1 high
            lohi = row['f2_hi'] - row['f1f2_hi']  # Only f2 high
            lolo = row['f1f2_lo']  # Both factors low
            
            counts_dict[timepoint] = np.array([hihi, hilo, lohi, lolo])
            print(f"Timepoint {timepoint}: HiHi={hihi}, HiLo={hilo}, LoHi={lohi}, LoLo={lolo}")
        return counts_dict
        
    def _simulate_for_params(self, transition_rates, initial_probs, n_sims=1000):
        """
        Run simulation with given parameters and return fractions.
        
        Args:
            transition_rates: Array of 8 transition probabilities between states
            initial_probs: Array of 3 initial state probabilities (4th inferred)
            n_sims: Number of simulations to run
            
        Returns:
            Dictionary mapping timepoints to arrays of state fractions
        """
        # Create TransitionParams object with rates
        params = TransitionParams(
            q00_01=transition_rates[0], q00_10=transition_rates[1],  # From 00 state
            q01_00=transition_rates[2], q01_11=transition_rates[3],  # From 01 state
            q10_00=transition_rates[4], q10_11=transition_rates[5],  # From 10 state
            q11_01=transition_rates[6], q11_10=transition_rates[7]   # From 11 state
        )
        
        # Set up initial state probabilities
        init_probs = {
            'state_00': initial_probs[0],
            'state_01': initial_probs[1],
            'state_10': initial_probs[2],
            'state_11': 1 - sum(initial_probs[:3])  # Remaining probability
        }
        
        print(f"\nRunning {n_sims} simulations with:")
        print(f"Transition rates: {transition_rates}")
        print(f"Initial probabilities: {init_probs}")
        
        # Run simulations for all timepoints
        results = run_simulations(params, init_probs, n_sims, 
                                list(self.observed_counts.keys()))
        
        # Calculate fractions for each state at each timepoint
        fractions = {}
        for div, sims in results.items():
            points = np.array(sims)
            fracs = np.array([
                np.mean((points[:,0] > self.threshold) & (points[:,1] > self.threshold)),   # HiHi
                np.mean((points[:,0] > self.threshold) & (points[:,1] <= self.threshold)),  # HiLo 
                np.mean((points[:,0] <= self.threshold) & (points[:,1] > self.threshold)),  # LoHi
                np.mean((points[:,0] <= self.threshold) & (points[:,1] <= self.threshold))  # LoLo
            ])
            fractions[div] = fracs
            print(f"Division {div} fractions: {fracs}")
            
        return fractions
    
    def run_inference(self, n_samples=1000):
        """
        Run MCMC inference to estimate transition rates and initial probabilities.
        
        Args:
            n_samples: Number of MCMC samples to draw
            
        Returns:
            PyMC trace object containing posterior samples
        """
        print(f"\nStarting MCMC inference with {n_samples} samples...")
        
        with pm.Model() as model:
            # Prior for transition rates (between 0 and 1)
            transition_rates = pm.Beta("transition_rates", alpha=1, beta=1, shape=8)
            
            # Prior for initial probabilities (Dirichlet distribution)
            initial_probs = pm.Dirichlet("initial_probs", a=np.ones(3))
            
            # Custom likelihood function
            def likelihood(transition_rates, initial_probs):
                """Calculate log likelihood of data given parameters"""
                fractions = self._simulate_for_params(transition_rates, initial_probs)
                
                # Calculate log likelihood
                ll = 0
                for div, counts in self.observed_counts.items():
                    probs = fractions[div]
                    ll += pm.logp(pm.Multinomial.dist(n=sum(counts), p=probs), counts)
                return ll
            
            # Add likelihood to model
            pm.Potential("likelihood", likelihood(transition_rates, initial_probs))
            
            print("Starting MCMC sampling...")
            # Run MCMC
            trace = pm.sample(n_samples, tune=500, cores=1)
            print("MCMC sampling complete")
            
        return trace
    
    def analyze_results(self, trace):
        """
        Analyze and visualize MCMC results.
        
        Args:
            trace: PyMC trace object from run_inference()
            
        Returns:
            summary: Arviz summary of posterior distributions
            params_dict: Dictionary of mean parameter estimates
        """
        print("\nAnalyzing MCMC results...")
        
        # Generate summary statistics
        summary = az.summary(trace)
        print("\nParameter summary:")
        print(summary)
        
        # Plot parameter distributions
        az.plot_trace(trace)
        print("\nTrace plots generated")
        
        # Calculate mean parameter estimates
        mean_rates = trace.posterior["transition_rates"].mean(dim=["chain", "draw"]).values
        mean_initials = trace.posterior["initial_probs"].mean(dim=["chain", "draw"]).values
        
        # Create dictionary of parameter estimates
        params_dict = {
            "q00_01": mean_rates[0],
            "q00_10": mean_rates[1],
            "q01_00": mean_rates[2],
            "q01_11": mean_rates[3],
            "q10_00": mean_rates[4],
            "q10_11": mean_rates[5],
            "q11_01": mean_rates[6],
            "q11_10": mean_rates[7],
            "init_00": mean_initials[0],
            "init_01": mean_initials[1],
            "init_10": mean_initials[2],
            "init_11": 1 - sum(mean_initials)
        }
        
        print("\nMean parameter estimates:")
        for param, value in params_dict.items():
            print(f"{param}: {value:.3f}")
        
        return summary, params_dict

