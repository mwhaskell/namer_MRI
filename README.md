README.md

These scripts and data provide example code for the method described in:

Haskell et al. 2019, "Network Accelerated Motion 
Estimation and Reduction (NAMER): Convolutional neural network guided 
retrospective motion correction using a separable motion model"

To start run namer_recon.m in MATLAB. This script requires that keras with
TensorFlow backend installed on your machine, as MATLAB will call a python
script to evaluate a CNN built using keras.

This example was most recently tested using matlab 2017b, and uses that 
versions's syntax for optimization settings. etc.

Key scripts

namer_recon.m - This script performs the separable cost function 
                version of the NAMER method (Eqn 3 in Haskell et al. 2019),
                and corresponds to the result shown in the bottom left of 
                Figure 4-B in the paper.

run_namer_cnn.py -  This script evaluates all of the patches for a given
                    input image and returns the output of the motion 
                    artifact detecting CNN.

train_namer_cnn.py- This script is provided as an example of how the
                    CNN was constructed and trained. The training data is 
                    not provided to run this script to save on space, but
                    can be shared by emailing melissa.w.haskell@gmail.com.




