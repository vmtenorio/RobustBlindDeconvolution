# Robust Blind Deconvolution over Graphs

This repository contains the code for the paper ''Robust Blind Deconvolution on Graphs with Graph Denoising'', by Victor M. Tenorio, Samuel Rey and Antonio G. Marques, which has been submitted to ICASSP'24. The abstract of the paper reads as follows:

> Blind deconvolution over graphs involves using (observed) output graph signals to obtain both the inputs (sources) as well as the filter that drives (models) the graph diffusion process. This is an ill-posed problem that requires additional assumptions, such as the sources being sparse, to be solvable. This paper addresses the blind deconvolution problem in the presence of imperfect graph information (IGI), where the observed graph is a _perturbed_ version of the (unknown) true graph. While IGI is arguably more the norm than the exception, the body of literature on this topic is relatively small, due in part to the fact that translating these perturbations to the standard graph processing tools (e.g. eigenvectors or polynomials of the graph) is a challenging endeavour. To address this, we propose an optimization-based estimator that solves the blind identification in the vertex domain, aims at estimating the inverse of the generating filter, and accounts explicitly for additive graph perturbations. Preliminary numerical experiments showcase the effectiveness and potential of the proposed algorithm.


The experiments in the paper are included in the `TestPerturbation.ipynb` jupyter notebook. The rest of the repository is organized as follows:

- `opt.py` contains the functions that implement the algorithms proposed in the paper.
- `alternatives.py` contains the implementation of the algorithms previously proposed in the literature.
- `data.py` contains the methods used to generate the data for the experiments.
- `utils` folder contains several Graph Signal Processing (GSP) utilities.
