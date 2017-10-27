TRAINING_PARAMS = dict(
    NORMALIZE_METHOD = "ZSCORE",             # How features should be normalized
    DESIRED_NUM_INSTANCES = 1000000,          # Specify max number of instances (None uses all instances)
    SPLIT_METHOD = "KFOLD",                  # One of "70/30" or "KFOLD"
    NUM_SPLITS = 10,                         # K in K fold cross validation
    LEARNING_RATE = 0.1,                     # Stepsize for gradient descent
    TRAINING_EPOCHS = 100                    # Number of iterations of gradient descent training
)

# Extract the data.zip folder to the working directory (same as this file) - should be automated
# Specify a list of features using the FEATURES field,
# If these features are to be ommited then set OMIT_FEATURES to True
# If they are to be the features used for the inputs, set OMIT_FEATURES to False

# Dataset 1 - SUM with noise
SUM_WITH_NOISE = dict(
    FILE_NAME = "SUM_With_Noise.csv",
    DELIMETER = ";",
    MAX_CHUNK = 500000,
    FEATURES = ["Instance", "Noisy Target", "Noisy Target Class"],
    OMIT_FEATURES = True,
    LABEL = "Noisy Target Class"
)

# Dataset 2 - SUM without noise
SUM_WITHOUT_NOISE = dict(
    FILE_NAME = "SUM_Without_Noise.csv",
    DELIMETER = ";",
    MAX_CHUNK = 500000,
    FEATURES = ["Instance", "Target", "Target Class"],
    OMIT_FEATURES = True,
    LABEL = "Target Class"
)

# Dataset 3 - 10k of SUM without noise (for faster debugging)
SUM_10K_WITHOUT_NOISE = dict(
    FILE_NAME = "sum_10k_without_noise.csv",
    DELIMETER = ";",
    MAX_CHUNK = 10000,
    FEATURES = ["Instance", "Target", "Target Class"],
    OMIT_FEATURES = True,
    LABEL = "Target Class"
)

 # Dataset 4 - House Data
HOUSE_DATA = dict(
    FILE_NAME = "kc_house_data.csv",
    DELIMETER = ",",
    FEATURES = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "price"],
    MAX_CHUNK = 20000,
    OMIT_FEATURES = False,
    LABEL = "condition"
)


###########################################
####                 KNN            #######
###########################################


ACTIVE_DATASET = HOUSE_DATA