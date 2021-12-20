# # Plots trajectories of a an input to a given model.

# import logging
# import torch
# from omegaconf import DictConfig
# from matplotlib import pyplot as plt
# import torch 
# from sklearn.decomposition import PCA

# # Library imports
# from core.models.base_model import TransductionModel

# log = logging.getLogger(__name__)

# class Trajectory:

#   def __init__(self):
#     pass

#   def plot_sequence(self, sequence, dim1, dim2):
#     dim1, dim1_idx = dim1
#     dim2, dim2_idx = dim2

#     plt.scatter(dim1, dim2)
#     head_width = (max(dim1) - min(dim1))/50
#     for i in range(len(dim1) - 1):
#       plt.arrow(dim1[i], dim2[i], dim1[i+1] - dim1[1], dim2[i+1] - dim2[i], head_width=head_width, length_includes_head=True)
#     for i, word in enumerate(sequence):
#       plt.text(dim1[i]+0.03, dim2[i]+0.03, word, fontsize=9)
    
#     plt.xlabel(f"Principal Component {dim1_idx+1}")
#     plt.xlabel(f"Principal Component {dim2_idx+1}")
#     plt.show()
