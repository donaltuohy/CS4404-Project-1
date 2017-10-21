import os


TRAINING_PARAMS = dict(
    NORMALIZE_METHOD = "SCALE",              # How features should be normalized
    DESIRED_NUM_INSTANCES = None,            # Specify max number of instances (None uses all instances)
    SPLIT_METHOD = "70/30",                  # One of "70/30" or "10Fold"
    LEARNING_RATE = 0.1,                     # Stepsize for gradient descent
    TRAINING_EPOCHS = 100                    # Number of iterations of gradient descent training
)


# Dataset 1 - SUM with noise
SUM_WITH_NOISE = dict(
    FILE_NAME = "SUM_with_noise.csv",
    OMITTED_FEATURES = ["Instance", "Noisy Target", "Noisy Target Class"],
    LABEL = "Noisy Target"
)

# Dataset 2 - House Data
HOUSE_DATA = dict(
    FILE_NAME = "SUM_with_noise.csv",
    OMITTED_FEATURES = ["Instance", "Noisy Target", "Noisy Target Class"],
    LABEL = "Noisy Target"
)

# Dataset 3 - 10k of SUM without noise (for faster debugging)
SUM_10K_WITHOUT_NOISE = dict(
    FILE_NAME = "SUM_10k_without_noise.csv",
    OMITTED_FEATURES = ["Instance", "Target", "Target Class"],
    LABEL = "Target"
)

ACTIVE_DATASET = SUM_10K_WITHOUT_NOISE
