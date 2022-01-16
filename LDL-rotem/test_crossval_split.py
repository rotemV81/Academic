import itertools
import numpy as np
import os

from conf import RANDOM_SEED, PATH_IMAGES, PATH_IMAGE_SETS, TRANSFORM_TS, TRANSFORM_TR
from dataset import dataset_processing

np.random.seed(RANDOM_SEED)

# Verify cross-validation split:
# Any two test parts do not share samples
for i, j in itertools.combinations(range(3), 2):

    image_set_i = os.path.join(PATH_IMAGE_SETS, f'NNEW_test_{i}.txt')
    test_data_i = dataset_processing.DatasetProcessing(PATH_IMAGES, image_set_i, transform=TRANSFORM_TS)

    image_set_j = os.path.join(PATH_IMAGE_SETS, f'NNEW_test_{j}.txt')
    test_data_j = dataset_processing.DatasetProcessing(PATH_IMAGES, image_set_j, transform=TRANSFORM_TS)

    assert [id for id in test_data_i.image_ids if id in test_data_j.image_ids]==[]


# Verify cross-validation split:
# Any test part does not share samples with its corresponding train part
for i in range(3):

    ts_set_i  = os.path.join(PATH_IMAGE_SETS, f'NNEW_test_{i}.txt')
    ts_data_i = dataset_processing.DatasetProcessing(PATH_IMAGES, ts_set_i, transform=TRANSFORM_TS)

    tr_set_i = os.path.join(PATH_IMAGE_SETS, f'NNEW_trainval_{i}.txt')
    tr_data_i = dataset_processing.DatasetProcessing(PATH_IMAGES, tr_set_i, transform=TRANSFORM_TR)

    assert [id for id in ts_data_i.image_ids if id in tr_data_i.image_ids]==[]
