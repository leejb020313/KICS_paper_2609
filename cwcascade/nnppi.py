"""NN-PPI baseline (Amatya et al., arXiv:2608.30731).

A small LM's raw score is corrected by the mean residual of its k nearest
labelled calibration neighbours:

    theta_i = c_i + (1/k) * sum_{j in S_i} (Y_j - c_j)

where S_i are the k calibration sentences with the highest cosine similarity
to sentence i. Embeddings must be L2-normalised so that a dot product is the
cosine similarity.
"""
import numpy as np


def nn_ppi(test_scores, test_emb, calib_scores, calib_emb, calib_labels, k):
    """Return the NN-PPI corrected scores theta for the test sentences."""
    sims = test_emb @ calib_emb.T
    neighbours = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
    residuals = calib_labels[neighbours] - calib_scores[neighbours]
    # sum of (1/k)-weighted residuals rather than .mean(): Gemma scores are coarse, so theta often
    # lands exactly on the 0.5 threshold and the last floating-point bit decides the label
    return test_scores + np.array([np.sum(np.full(k, 1.0 / k) * r) for r in residuals])
