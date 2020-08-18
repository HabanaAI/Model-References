## PyTorch Recommendation Models for Gaudi

This page will contain a general description on how to use and optmize Recommendation models for PyTorch on Gaudi. 

Deep learning recommendation model (DLRM) model included in the docker is based on Facebook’s research git repository - https://github.com/facebookresearch/dlrm.

The demo_dlrm is a wrapper script for dlrm_s_pytorch*.py script. These scripts are available in Model-References/PyTorch/recommendation/Dlrm folder. The model has been modified to support Habana devices and to make use of custom extension operators. 