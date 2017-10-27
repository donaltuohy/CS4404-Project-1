TRAINING_PARAMS = dict(
    NORMALIZE_METHOD = "ZSCORE",             # How features should be normalized
    DESIRED_NUM_INSTANCES = 100000,          # Specify max number of instances (None uses all instances)
    SPLIT_METHOD = "70/30",                  # One of "70/30" or "KFOLD"
    NUMBER_NEIGHBORS = 2,                   # Number of neighbors in KNN 
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
    FEATURES = ["Instance", "Noisy Target", "Noisy Target Class"],
    OMIT_FEATURES = True,
    LABEL = "Noisy Target"
)

# Dataset 2 - House Data
HOUSE_DATA = dict(
    FILE_NAME = "kc_house_data.csv",
    DELIMETER = ",",
    FEATURES = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot"],
    OMIT_FEATURES = False,
    LABEL = "price"
)


# Dataset 3 - 10k of SUM without noise (for faster debugging)
SUM_10K_WITHOUT_NOISE = dict(
    FILE_NAME = "sum_10k_without_noise.csv",
    DELIMETER = ";",
    FEATURES = ["Instance", "Target", "Target Class"],
    OMIT_FEATURES = True,
    LABEL = "Target"
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
###########################################
HOUSE_DATA_KNN = dict(HOUSE_DATA)
#HOUSE_DATA_KNN['FEATURES'] = ['price', 'bathrooms', 'sqft_living', 'sqft_lot', 'condition', 'grade', 'waterfront', 'view', 'sqft_above', 'sqft_basement']

# Great results for house data > 90% accuracy
HOUSE_DATA_KNN['FEATURES'] = ['condition','id', 'date', 'floors', 'yr_built', 'yr_renovated', 'zipcode']
HOUSE_DATA_KNN['OMIT_FEATURES'] = True
HOUSE_DATA_KNN['LABEL'] = "condition"


ACTIVE_DATASET = HOUSE_DATA_KNN