import numpy as np
import settings
import databuilder

# The npy data set file
DATA_SET_PATH: str = settings.saved_npy_data_set_path
# Data preprocessor. Set to True (by default) if need to change something in training data.
REBUILD_DATA: bool = settings.rebuild_data_set_flag

if REBUILD_DATA:
    data_build = databuilder.DataBuilder()
    data_build.make_training_data()

training_data = np.load(DATA_SET_PATH, allow_pickle=True)

print("Number of samples: ", len(training_data))

print("DATA PREPROCESSED")
