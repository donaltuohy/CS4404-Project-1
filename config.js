const TRAINING_PARAMS = {
    NORMALIZE_METHOD: "ZSCORE",             // How features should be normalized
    DESIRED_NUM_INSTANCES: 100000,          // Specify max number of instances (null uses all instances)
    SPLIT_METHOD: "70/30",                  // One of "70/30" or "KFOLD".
    NUM_FOLDS: 10,                          // Number of folds
    LEARNING_RATE: 0.1,                     // Stepsize for gradient descent
    TRAINING_EPOCHS: 100,                   // Number of iterations of gradient descent training
    K: 2,                                   // Number of nearest neighbours 
    KNN_THRESHOLD: 0                        // Allowable deviation from actual value in KNN numeric clasification
};

/*
Extract the data.zip folder to the working directory (same as this file) - should be automated
Specify a list of features using the FEATURES field,
If these features are to be ommited then set OMIT_FEATURES to True
If they are to be the features used for the inputs, set OMIT_FEATURES to False
*/


const LINEAR_REGRESSION_DATASETS = {
    // Dataset 1 - SUM with noise
    SUM_WITH_NOISE: {
        FILE_NAME: "sum_with_noise.csv",
        DELIMETER: ";",
        FEATURES: ["Instance", "Noisy Target", "Noisy Target Class"],
        OMIT_FEATURES: true,
        LABEL: "Noisy Target"
    },

    // Dataset 2 - House Data
    HOUSE_DATA: {
        FILE_NAME: "kc_house_data.csv",
        DELIMETER: ",",
        FEATURES: ["bedrooms", "bathrooms", "sqft_living", "sqft_lot"],
        OMIT_FEATURES: false,
        LABEL: "price"
    },

    // Dataset 3 - 10k of SUM without noise (for faster debugging)
    SUM_10K_WITHOUT_NOISE: {
        FILE_NAME: "sum_10k_without_noise.csv",
        DELIMETER: ",",
        FEATURES: ["Instance", "Target", "Target Class"],
        OMIT_FEATURES: true,
        LABEL: "Target"
    },

    // Dataset 4 - 10k of SUM without noise (for faster debugging)
    SUM_10K_WITH_NOISE: {
        FILE_NAME: "sum_10k_with_noise.csv",
        DELIMETER: ";",
        FEATURES: ["Instance", "Noisy Target", "Noisy Target Class"],
        OMIT_FEATURES: true,
        LABEL: "Noisy Target"
    },
};

const ACTIVE_LINEAR_REGRESSION_DATASET = LINEAR_REGRESSION_DATASETS['HOUSE_DATA'];



/******************************
    KNN
*******************************/
const SUM_WITH_NOISE_KNN = Object.assign({}, LINEAR_REGRESSION_DATASETS['SUM_WITH_NOISE'])
SUM_WITH_NOISE_KNN['LABEL'] = "Noisy Target"

// const HOUSE_DATA_KNN = Object.assign({}, LINEAR_REGRESSION_DATASETS['HOUSE_DATA']);
// HOUSE_DATA_KNN['FEATURES'] = ["LotArea", "OverallQual", "OverallCond", "SalePrice", "GrLivArea", "FullBath"]
// HOUSE_DATA_KNN['LABEL'] = "BedroomAbvGr"

const HOUSE_DATA_KNN = Object.assign({}, LINEAR_REGRESSION_DATASETS['HOUSE_DATA']);
HOUSE_DATA_KNN['FEATURES'] = ['id', 'date', 'floors', 'yr_built', 'yr_renovated', 'zipcode', 'condition']
HOUSE_DATA_KNN['OMIT_FEATURES'] = true
HOUSE_DATA_KNN['LABEL'] = "condition"


const SUM_10K_WITHOUT_NOISE_KNN = Object.assign({}, LINEAR_REGRESSION_DATASETS['SUM_10K_WITHOUT_NOISE'])
SUM_10K_WITHOUT_NOISE_KNN['LABEL'] = "Target"

const KNN_DATASETS = {
    SUM_WITH_NOISE_KNN,
    SUM_10K_WITHOUT_NOISE_KNN,
    HOUSE_DATA_KNN
};

const ACTIVE_KNN_DATASET = KNN_DATASETS['HOUSE_DATA_KNN'];

module.exports = {
    ACTIVE_LINEAR_REGRESSION_DATASET,
    ACTIVE_KNN_DATASET,
    TRAINING_PARAMS
}