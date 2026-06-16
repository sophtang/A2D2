from transformers import AutoModelForMaskedLM
import numpy as np
from tdc import Oracle, Evaluator

class MolScoringFunctions:
    def __init__(self, score_func_names=None, device=None, sa_transform='inverse'):
        """
        Class for generating score vectors given generated sequence

        Args:
            score_func_names: list of scoring function names to be evaluated
            score_weights: weights to scale scores (default: 1)
            sa_transform: how to transform SA scores to higher-is-better ~[0,1]:
                'inverse' (default): 1/(1+SA)  — range ~0.09-0.5, weak gradient
                'linear':  (10-SA)/9            — range ~0-1, stronger gradient
        """
        if score_func_names is None:
            # just do unmasking based on validity of peptide bonds
            self.score_func_names = []
        else:
            self.score_func_names = score_func_names
        
        self.sa_transform = sa_transform

        oracle_qed = Oracle('qed')
        oracle_sa = Oracle('sa')

        self.all_funcs = {'qed': oracle_qed,
                          'sa': oracle_sa,
                          } 
        
    def forward(self, input_seqs):
        scores = []
        
        for i, score_func in enumerate(self.score_func_names): 
            score = self.all_funcs[score_func](input_seqs)
            
            # Transform SA to be maximized and normalized (original SA: 1-10, lower is better)
            # Convert to: higher is better, normalized to ~0-1 range like QED
            if score_func == 'sa':
                if self.sa_transform == 'linear':
                    score = (10.0 - np.array(score)) / 9.0  # range ~0-1, clipped at 0
                    score = np.maximum(score, 0.0)
                else:
                    score = 1.0 / (1.0 + np.array(score))  # range ~0.09-0.5
            
            scores.append(score)
            
        # convert to numpy arrays with shape (num_sequences, num_functions)
        scores = np.float32(scores).T
        
        return scores
    
    def __call__(self, input_seqs: list):
        return self.forward(input_seqs)


def unittest():
    scoring = MolScoringFunctions(score_func_names=['qed', 'sa'])
    
    smiles = ['CCOc1cc(ccc1NC(=O)N[C@@H]2CCCC[C@@H]2O)F']
    
    scores = scoring(input_seqs=smiles)
    print(scores)
    print(len(scores))

if __name__ == '__main__':
    unittest()