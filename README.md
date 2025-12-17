Python code for reproducing the results from the paper "*Balancing Accuracy and Speed: A Multi-Fidelity
Ensemble Kalman Filter with a Machine Learning
Surrogate Model*" by Jeffrey van der Voort, Martin Verlaan and Hanne Kekkonen. Available on Arxiv: https://arxiv.org/abs/2512.12276. Please cite the paper when using the code.

This repository consists of 2 folders: Lorenz-2005 and QG model, which contain the files to reproduce the experiments for each test model.

**Lorenz-2005 folder**

Contents:
- MF_EnKF_Lorenz2005.ipynb     this file 

**QG model folder**

Contents:
- **MF_EnKF_QGmodel.ipynb**.     Main file, from here you can run the MF-EnKF experiments calling on the other files in the folder
- **EnKF.py**.                    Python file containing the Ensemble Kalman Filter (EnKF) code.
- **MF_EnKF.py**.                   Python file containing the Multi-Fidelity EnKF (MF-EnKF) code.
- **QG.py**.                        Python file containing the numerical solver for the Quasi-Geostrophic model equations.
- **QG_surrogate.py**.              Python file containing the ML surrogate model for the QG model.
- **QGmodel_weights_Unet**.         Weights of the QG ML surrogate model, obtained after training
- **baseline.py**.                  Python file containing baseline method used as comparison in the paper
- **generate_obs_and_truth.py**.    Python file to generate the truth trajectory psi_true and the observations
- **intializeQG.py**.               Python file to generate the initial condition and ensemble for the experiments
- **localization.py**.              Python file containing the localization functions
