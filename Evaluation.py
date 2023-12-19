import torch
import numpy as np
import cv2
from function import *
from network import *

img_size = 224
num_class = 21

def compute_mean_iou(pred, target, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        iou = intersection / (union + 1e-10)
        if iou > 0.0:
            ious.append(iou)

    return np.mean(ious)


def evaluate_model(model, test_img, test_gt, img_size, num_classes):
    model.eval()
    num_samples = test_img.shape[0]
    mIOU_list = []

    for itest in range(num_samples):
        img_temp = test_img[itest:itest + 1, :, :, :].astype(np.float32)
        img_temp = (img_temp / 255.0) * 2 - 1  # [1, 28, 28]
        img_temp = np.transpose(img_temp, (0, 3, 1, 2))

        with torch.no_grad():
            pred = model(torch.from_numpy(img_temp.astype(np.float32)))

        pred = pred.cpu().numpy()
        pred = np.argmax(pred[0, :, :, :], axis=0)
        pred = pred[:, :, np.newaxis]

        gt_mask = test_gt[itest, :, :, 1].astype(np.int64)

        pred_resized = cv2.resize(pred, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

        mIOU = compute_mean_iou(pred_resized, gt_mask, num_classes)
        mIOU_list.append(mIOU)

    average_mIOU = np.mean(mIOU_list)

    return average_mIOU

# Path
model_path = "/home/hoya9802/PycharmProjects/pythonProject/torchenv/FCRN/DFCRN_model/model_40000.pt"
path = "/home/hoya9802/Downloads/VOC_dataset/"

# models
model = FCN_8S(num_class)
# model = FCRN_8S(num_class)
# model = DFCRN_8S(num_class)

model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Without Canny Edge
test_img, test_gt = load_semantic_seg_data(path + 'test/test_img/', path + 'test/test_gt/', img_size=img_size)

# With Canny Edge
# test_img, test_gt = load_semantic_seg_data_canny(path + 'test/test_img/', path + 'test/test_gt/', path + 'test/test_ce/', img_size=img_size)

average_mIOU = evaluate_model(model, test_img, test_gt, img_size, num_class)

print("Mean IOU on the test dataset: {:.4f}".format(average_mIOU))
