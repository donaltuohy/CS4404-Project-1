TRAINING_PARAMS = dict(
    NORMALIZE_METHOD = "ZSCORE",             # How features should be normalized
    DESIRED_NUM_INSTANCES = 10000000,          # Specify max number of instances (None uses all instances)
    SPLIT_METHOD = "70/30",                  # One of "70/30" or "10Fold"
    LEARNING_RATE = 0.1,                     # Stepsize for gradient descent
    TRAINING_EPOCHS = 100                    # Number of iterations of gradient descent training
)

# Extract the data.zip folder to the working directory (same as this file) - should be automated
# Specify a list of features using the FEATURES field,
# If these features are to be ommited then set OMIT_FEATURES to True
# If they are to be the features used for the inputs, set OMIT_FEATURES to False

# Dataset 1 - SUM with noise
SUM_WITH_NOISE = dict(
    FILE_NAME = "The SUM dataset, without noise.csv",
    DELIMETER = ";",
    FEATURES = ["Instance", "Target" ,"Target Class"],
    OMIT_FEATURES = True,
    LABEL = "Target"
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
    DELIMETER = ",",
    FEATURES = ["Instance", "Target", "Target Class"],
    OMIT_FEATURES = True,
    LABEL = "Target"
)

# Dataset 4 - 10k of SUM with noise (for faster debugging)
SUM_10K_WITH_NOISE = dict(
    FILE_NAME = "sum_10k_with_noise.csv",
    DELIMETER = ",",
    FEATURES = ["Instance", "Noisy Target", "Noisy Target Class"],
    OMIT_FEATURES = True,
    LABEL = "Noisy Target"
)


TEST_CLASSIFICATION_DATA = dict(
    FILE_NAME = "test_classification.csv",
    DELIMETER = ",",
    FEATURES = ["x", "y"],
    OMIT_FEATURES = False,
    LABEL = "class"   
)


###########################################
####                 KNN            #######
#########################################

"""

SUM_WITH_NOISE_KNN = dict(SUM_WITH_NOISE)
SUM_WITH_NOISE_KNN['LABEL'] = "Noisy Target Class"

SUM_10K_WITHOUT_NOISE_KNN = dict(SUM_10K_WITHOUT_NOISE)
SUM_10K_WITHOUT_NOISE_KNN['LABEL'] = "Target Class"


HOUSE_DATA_KNN = dict(HOUSE_DATA)
HOUSE_DATA_KNN['FEATURES'] = ["LotArea", "OverallQual", "OverallCond", "SalePrice", "GrLivArea", "FullBath"]
HOUSE_DATA_KNN['LABEL'] = "BedroomAbvGr"
"""

ACTIVE_DATASET = SUM_WITH_NOISE
