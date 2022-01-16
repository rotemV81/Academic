from dataset import dataset_processing
import numpy as np
import os
import time
import torch
from utils.report import report_metrics, report_mae_mse
from utils.utils import Logger
import torch.nn as nn
import warnings
from torch.utils.data import DataLoader
from conf import PATH_MODEL, PATH_IMAGES, RANDOM_SEED, PATH_IMAGE_SETS, TRANSFORM_TS, BATCH_SIZE_TS, NUM_WORKERS

np.random.seed(RANDOM_SEED)

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

saved_model = np.load(PATH_MODEL, allow_pickle=True)
saved_model = saved_model['cnn']

cnn = saved_model.item()

loss_func = nn.CrossEntropyLoss().cuda()

fold_idx = 0

with torch.no_grad():

    # Ground truth severity and lesion count
    lesions_true  = np.array([])
    severity_true = np.array([])

    # Predicted severity and lesion count
    lesions_pred  = np.array([])
    severity_pred = np.array([])

    # Also compare the ground truth severity to the sum of:
    # - severity distribution by global assessment, and
    # - severity distribution from lesion count
    severity_pred_max = np.array([])

    # Set the model to evaluation
    cnn.eval()

    test_set    = os.path.join(PATH_IMAGE_SETS, f'NNEW_test_{fold_idx}.txt')
    test_data   = dataset_processing.DatasetProcessing(PATH_IMAGES, test_set, transform=TRANSFORM_TS)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE_TS, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    for step, (test_images, test_classes, test_lesions) in enumerate(test_loader):

        test_images = test_images.cuda()
        test_classes = test_classes.cuda()  # Severity classes

        severity_true = np.hstack((severity_true, test_classes.data.cpu().numpy()))
        lesions_true = np.hstack((lesions_true, test_lesions.data.cpu().numpy()))

        cnn.eval() # TODO: is this required in each iteration?

        cls, cnt, cnt2cls = cnn(test_images, None)

        _, preds_m = torch.max(cls + cnt2cls, 1)
        _, preds = torch.max(cls, 1)

        y_pred = np.hstack((y_pred, preds.data.cpu().numpy()))
        y_pred_m = np.hstack((y_pred_m, preds_m.data.cpu().numpy()))

        _, preds_l = torch.max(cnt, 1)
        preds_l = (preds_l + 1).data.cpu().numpy()
        lesion_pred = np.hstack((lesion_pred, preds_l))

    # Sensitivity and specificity report
    _, _, report = report_metrics(y_pred, severity_true)
    print(str(report) + '\n')

    _, _, report_m = report_metrics(y_pred_m, severity_true)
    print(str(report_m) + '\n')

    _, _, _, mae_mse_report = report_mae_mse(lesions_true, lesion_pred, severity_true)
    print(str(mae_mse_report) + '\n')


# cross_val_lists = ['0', '1', '2', '3', '4']
# for cross_val_index in cross_val_lists:
#     log.write(f'cross_val_index: {cross_val_index}\n')
#     cnn = trainval_test(cross_val_index, sigma=30 * 0.1, lamda=6 * 0.1)
#     np.savez(f'/host_root/home/rotem/Private/Academic/LDL-rotem/trained_model.npz', cnn=cnn)

