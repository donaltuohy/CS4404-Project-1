TRAINING_PARAMS = dict(
    NORMALIZE_METHOD = "ZSCORE",             # How features should be normalized
    DESIRED_NUM_INSTANCES = None,            # Specify max number of instances (None uses all instances)
    SPLIT_METHOD = "70/30",                  # One of "70/30" or "10Fold"
    LEARNING_RATE = 0.1,                     # Stepsize for gradient descent
    TRAINING_EPOCHS = 100                    # Number of iterations of gradient descent training
)


# Specify a list of features using the FEATURES field,
# If these features are to be ommited then set OMIT_FEATURES to True
# If they are to be the features used for the inputs, set OMIT_FEATURES to False

# Dataset 1 - SUM with noise
SUM_WITH_NOISE = dict(
    FILE_NAME = "sum_with_noise.csv",
    DELIMETER = ";",
    FEATURES = ["Instance", "Noisy Target", "Noisy Target Class"],
    OMIT_FEATURES = True,
    LABEL = "Noisy Target"
)

# Dataset 2 - House Data
HOUSE_DATA = dict(
    FILE_NAME = "housing_data.csv",
    DELIMETER = ",",
    FEATURES = ["LotArea", "OverallQual", "OverallCond", "BedroomAbvGr"],
    OMIT_FEATURES = False,
    LABEL = "SalePrice"
)

# Dataset 3 - 10k of SUM without noise (for faster debugging)
SUM_10K_WITHOUT_NOISE = dict(
    FILE_NAME = "sum_10k_without_noise.csv",
    DELIMETER = ";",
    FEATURES = ["Instance", "Target", "Target Class"],
    OMIT_FEATURES = True,
    LABEL = "Target"
)

ACTIVE_DATASET = HOUSE_DATA
