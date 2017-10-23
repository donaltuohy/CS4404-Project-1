const TRAINING_PARAMS = {
    NORMALIZE_METHOD: "ZSCORE",             // How features should be normalized
    DESIRED_NUM_INSTANCES: 100000,          // Specify max number of instances (null uses all instances)
    SPLIT_METHOD: "70/30",                  // One of "70/30" or "10Fold".
    LEARNING_RATE: 0.1,                     // Stepsize for gradient descent
    TRAINING_EPOCHS: 100                    // Number of iterations of gradient descent training
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
        FILE_NAME: "housing_data.csv",
        DELIMETER: ",",
        FEATURES: ["LotArea", "OverallQual", "OverallCond", "BedroomAbvGr"],
        OMIT_FEATURES: false,
        LABEL: "SalePrice"
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
        DELIMETER: ",",
        FEATURES: ["Instance", "Noisy Target", "Noisy Target Class"],
        OMIT_FEATURES: true,
        LABEL: "Noisy Target"
    },

    // Dataset 5 - Test
    TEST_LINEAR_REGRESSION_DATA: {
        FILE_NAME: "test.csv",
        DELIMETER: ",",
        FEATURES: ["x1", "x2", "x3"],
        OMIT_FEATURES: false,
        LABEL: "y"   
    }
};

const ACTIVE_LINEAR_REGRESSION_DATASET = LINEAR_REGRESSION_DATASETS['SUM_10K_WITH_NOISE'];



/******************************
    KNN
*******************************/
const SUM_WITH_NOISE_KNN = Object.assign({}, LINEAR_REGRESSION_DATASETS['SUM_WITH_NOISE'])
SUM_WITH_NOISE_KNN['LABEL'] = "Noisy Target Class"


const HOUSE_DATA_KNN = Object.assign({}, LINEAR_REGRESSION_DATASETS['HOUSE_DATA']);
HOUSE_DATA_KNN['FEATURES'] = ["LotArea", "OverallQual", "OverallCond", "SalePrice", "GrLivArea", "FullBath"]
HOUSE_DATA_KNN['LABEL'] = "BedroomAbvGr"

const KNN_DATASETS = {
    SUM_WITH_NOISE_KNN,
    HOUSE_DATA_KNN
};

const ACTIVE_KNN_DATASET = KNN_DATASETS['SUM_WITH_NOISE_KNN'];

module.exports = {
    ACTIVE_LINEAR_REGRESSION_DATASET,
    ACTIVE_KNN_DATASET,
    TRAINING_PARAMS
}