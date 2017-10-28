TRAINING_PARAMS = dict(
    NORMALIZE_METHOD = "ZSCORE",            # How features should be normalized
    DESIRED_NUM_INSTANCES = 100000,         # Specify max number of instances (None uses all instances)
    SPLIT_METHOD = "KFOLD",                 # One of "70/30" or "KFOLD"
    NUM_SPLITS = 10,                        # K in K fold cross validation
    LEARNING_RATE = 0.1,                    # Stepsize for gradient descent
    TRAINING_EPOCHS = 100,                  # Number of iterations of gradient descent training
    
    IS_KNN_LABEL_STRING = False,            # If predicted string categorical data, set to True
    KNN_CLASS_THRESHOLD = None,             # The accepted deviation from true y value for numeric classification                                # Can be None for exact classification
    K = 2                                   # Number of nearest neighbours to use
)


# Specify a list of features using the FEATURES field,
# If these features are to be ommited then set OMIT_FEATURES to True
# If they are to be the features used for the inputs, set OMIT_FEATURES to False

# Dataset 1 - SUM with noise
SUM_WITH_NOISE = dict(
    FILE_NAME = "sum_100k_with_noise.csv",
    DELIMETER = ";",
    FEATURES = ["Instance", "Noisy Target", "Noisy Target Class"],
    OMIT_FEATURES = True,
    LABEL = "Noisy Target"
)

# Dataset 2 - House Data
# Attempting to predict house prices based on number of bathrooms, bedrooms, size of living and size of lot
HOUSE_DATA = dict(
    FILE_NAME = "kc_house_data.csv",
    DELIMETER = ",",
    FEATURES = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot"],
    OMIT_FEATURES = False,
    LABEL = "price"
)


# Dataset 3 - 10k of SUM without noise (for faster debugging/testing)
SUM_10K_WITHOUT_NOISE = dict(
    FILE_NAME = "sum_10k_without_noise.csv",
    DELIMETER = ",",
    FEATURES = ["Instance", "Target", "Target Class"],
    OMIT_FEATURES = True,
    LABEL = "Target"
)


###########################################
####                 KNN            #######
###########################################
SUM_WITH_NOISE_KNN = dict(SUM_WITH_NOISE)
SUM_WITH_NOISE_KNN['LABEL'] = "Noisy Target"

# Attempting to predict condition of a house using all features but the ones below
HOUSE_DATA_KNN = dict(HOUSE_DATA)
HOUSE_DATA_KNN['FEATURES'] = ['id', 'date', 'floors', 'yr_built', 'yr_renovated', 'zipcode', 'condition']
HOUSE_DATA_KNN['OMIT_FEATURES'] = True
HOUSE_DATA_KNN['LABEL'] = "condition"


# Handy for quick testing
SUM_10K_WITHOUT_NOISE_KNN = dict(SUM_10K_WITHOUT_NOISE)
SUM_10K_WITHOUT_NOISE_KNN['LABEL'] = "Target"

# Select the dataset to be used by the algorithm
# Be sure to sure to use the <>_KNN datasets when using K nearest neighbours 
# then simply run `python <algorithm-script.py>` to get the results.
ACTIVE_DATASET = HOUSE_DATA_KNN

