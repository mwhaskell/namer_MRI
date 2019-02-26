# ------------------------------------------------------------------------------
#
# run_namer_cnn.py
#
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# import libraries
# ------------------------------------------------------------------------------
import os
import scipy.io as sio
import numpy as np
from keras.models import load_model
import tensorflow
from keras.backend.tensorflow_backend import set_session
import sys


# ------------------------------------------------------------------------------
# script initialization
# ------------------------------------------------------------------------------

# get filenames and variable names from input arguments to python call
var = sys.argv
var = var[1:]

in_filename = var[0]
in_var = var[1]
out_filename = var[2]
out_var = var[3]
model_filename = var[4]
gpu_str = var[5]

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# The GPU id to use, usually either "0" or "1"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str

# set tensorflow environment to limit GPU memory usage, and select GPU
config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tensorflow.Session(config=config))

# initialize paths and filenames (fn abbreviation) and variable names (vn abbreviation)
mat_data = sio.loadmat(in_filename)
test_patches = mat_data[in_var]
print("Data structures loaded...")


# ------------------------------------------------------------------------------
# reload cnn to view results
# ------------------------------------------------------------------------------

model = load_model(model_filename)
test_patches_model_output = model.predict(test_patches, batch_size=2500)
print("Patches evaluated. Saving...")

sio.savemat(out_filename, {out_var: test_patches_model_output})


