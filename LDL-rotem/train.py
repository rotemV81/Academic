import numpy as np
import os
import random
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from timeit import default_timer as timer
import warnings

from utils.genLD import gen_label_distribution
from utils.report import report_metrics, report_mae_mse
from utils.utils import Logger, AverageMeter, time_to_str

from dataset import dataset_processing
from conf import TRANSFORM_TS, TRANSFORM_TR, PATH_IMAGES, BATCH_SIZE_TR, BATCH_SIZE_TS
from model.resnet50 import resnet50

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# Set random seed for reproducibility
np.random.seed(42)

# Hyper Parameters
LR              = 0.001                 # learning rate
NUM_WORKERS     = 6
lr_steps        = [30, 60, 90, 120]     # adjust the learning rate at these epoch

LOG_DIR   = '/host_root/home/rotem/Private/Academic/LDL-rotem/logs'
time_now  = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
LOG_NAME  = f'log_{time_now}.log'
LOG_PATH  = os.path.join(LOG_DIR, LOG_NAME)

log = Logger()
log.open(LOG_PATH)


def adjust_learning_rate_new(optimizer, decay=0.5):
    """ Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs """
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


def trainval_test(cross_val_index, sigma, lamda):

    TRAIN_FILE = '/host_root/fastspace/datasets/external/Acne04/Detection/VOC2007/ImageSets/Main/NNEW_trainval_' + cross_val_index + '.txt'
    dset_train = dataset_processing.DatasetProcessing(PATH_IMAGES, TRAIN_FILE, transform=TRANSFORM_TR)
    train_loader = DataLoader(dset_train, batch_size=BATCH_SIZE_TR, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)


    TEST_FILE = '/host_root/fastspace/datasets/external/Acne04/Detection/VOC2007/ImageSets/Main/NNEW_test_' + cross_val_index + '.txt'
    dset_test = dataset_processing.DatasetProcessing(PATH_IMAGES, TEST_FILE, transform=TRANSFORM_TS)
    test_loader = DataLoader(dset_test, batch_size=BATCH_SIZE_TS, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    cnn = resnet50().cuda()

    cudnn.benchmark = True

    params = []
    for key, value in dict(cnn.named_parameters()).items():
        if value.requires_grad:
            if any(name in key for name in ['fc', 'counting']):
                params += [{'params': [value], 'lr': LR * 1.0, 'weight_decay': 5e-4}]
            else:
                params += [{'params': [value], 'lr': LR * 1.0, 'weight_decay': 5e-4}]

    optimizer = torch.optim.SGD(params, momentum=0.9)

    loss_func = nn.CrossEntropyLoss().cuda()
    kl_loss_1 = nn.KLDivLoss().cuda()
    kl_loss_2 = nn.KLDivLoss().cuda()
    kl_loss_3 = nn.KLDivLoss().cuda()

    # training and testing
    start = timer()

    for epoch in range(lr_steps[-1]):  # EPOCH

        if epoch in lr_steps:
            adjust_learning_rate_new(optimizer, 0.5)

        losses_cls     = AverageMeter()
        losses_cnt     = AverageMeter()
        losses_cnt2cls = AverageMeter()
        losses         = AverageMeter()

        # [rotem:] the authors override the original resnet50 train to set the training mode as desired
        cnn.train()

        for step, (batch_images, batch_classes, batch_lesions) in tqdm(enumerate(train_loader)):   # gives batch data, normalize x when iterate train_loader

            # Generate lesion count distribution per batch sample
            lesion_count = gen_label_distribution(batch_lesions.numpy() - 1, sigma, 'klloss', 65)

            # Split each distribution according to severity levels (see fig. 1-d in the paper)
            severity_from_count = np.vstack((np.sum(lesion_count[:, :5], 1),                    # 1-4: mild
                                             np.sum(lesion_count[:, 5:20], 1),                  # 5-19: moderate
                                             np.sum(lesion_count[:, 20:50], 1),                 # 20-49: severe
                                             np.sum(lesion_count[:, 50:], 1))).transpose()      # 50-65: very severe

            lesion_count = torch.from_numpy(lesion_count).cuda().float()
            severity_from_count = torch.from_numpy(severity_from_count).cuda().float()

            # overrides model's 'train' method to set the training mode as desired
            cnn.train()     # TODO: can this be taken outside the loop?

            batch_images = batch_images.cuda()

            # feed forward
            cls, cnt, cnt2cls = cnn(batch_images, None)

            loss_cls     = kl_loss_1(torch.log(cls), severity_from_count) * 4.0
            loss_cnt     = kl_loss_2(torch.log(cnt), lesion_count) * 65.0
            loss_cnt2cls = kl_loss_3(torch.log(cnt2cls), severity_from_count) * 4.0

            loss = (loss_cls + loss_cnt2cls) * 0.5 * lamda + loss_cnt * (1.0 - lamda)

            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

            # update tracking variables
            losses_cls.update(loss_cls.item(), batch_images.size(0))
            losses_cnt.update(loss_cnt.item(), batch_images.size(0))
            losses_cnt2cls.update(loss_cnt2cls.item(), batch_images.size(0))
            losses.update(loss.item(), batch_images.size(0))

        elapsed = time_to_str((timer() - start))
        message = f'epoch {epoch} | {losses_cls.avg:.3f} | {losses_cnt.avg:.3f} | {losses_cnt2cls.avg:.3f} | {losses.avg:.3f} | {elapsed} \n'
        log.write(message)

        # Evaluate after each epoch starting from epoch number 9
        if epoch >= 0:

            with torch.no_grad():
                test_loss     = 0
                severity_hits = 0

                true_classes = np.array([])
                pred_classes = np.array([])

                true_counts = np.array([])
                pred_counts = np.array([])

                pred_classes_m = np.array([])

                # Sets the model in evaluation mode
                cnn.eval()

                for step, (batch_images, batch_classes, batch_counts) in enumerate(test_loader):   # gives batch data, normalize x when iterate train_loader

                    batch_images  = batch_images.cuda()
                    batch_classes = batch_classes.cuda()

                    true_classes = np.hstack((true_classes, batch_classes.data.cpu().numpy()))
                    true_counts = np.hstack((true_counts, batch_counts.data.cpu().numpy()))

                    cnn.eval() # TODO: is this required in each iteration?

                    cls, cnt, cnt2cls = cnn(batch_images, None)

                    loss = loss_func(cnt2cls, batch_classes)
                    test_loss += loss.data

                    _, pred_class   = torch.max(cls, 1)             # predicted severities
                    _, pred_count   = torch.max(cnt, 1)             # Predicted lesion counts
                    _, pred_class_m = torch.max(cls + cnt2cls, 1)   # predicted severities based on sum of severity estimate and severity-from-count

                    pred_classes    = np.hstack((pred_classes,   pred_class.data.cpu().numpy()))
                    pred_counts     = np.hstack((pred_counts,    (pred_count + 1).data.cpu().numpy()))
                    pred_classes_m  = np.hstack((pred_classes_m, pred_class_m.data.cpu().numpy()))

                    severity_hits += torch.sum((pred_class == batch_classes)).data.cpu().numpy()

                _, _, report = report_metrics(pred_classes, true_classes)
                log.write(str(report) + '\n')

                _, _, report_m = report_metrics(pred_classes_m, true_classes)
                log.write(str(report_m) + '\n')

                _, _, _, mae_mse_report = report_mae_mse(true_counts, pred_counts, true_classes)
                log.write(str(mae_mse_report) + '\n')

    return cnn


cross_val_lists = ['0', '1', '2', '3', '4']

for cross_val_index in cross_val_lists:
    log.write(f'cross_val_index: {cross_val_index}\n')
    cnn = trainval_test(cross_val_index, sigma=30 * 0.1, lamda=6 * 0.1)
    # np.savez(f'/host_root/home/rotem/Private/Academic/LDL-rotem/trained_model.npz', cnn=cnn)
