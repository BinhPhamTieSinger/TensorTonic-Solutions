import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    predictions = np.array(predictions)
    num_samples = predictions.shape[1]
    
    votes = []
    
    for col in range(num_samples):
        sample_preds = predictions[:, col]
        values, counts = np.unique(sample_preds, return_counts=True)
        
        max_count = counts.max()
        candidates = values[counts == max_count]
        votes.append(candidates.min())
    
    return votes