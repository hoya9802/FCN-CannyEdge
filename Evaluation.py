import numpy as np
import torch
from function import load_semantic_seg_data
from network import FCRN_8S
from sklearn.metrics import confusion_matrix

Device = torch.device("mps")
print(Device)

num_class = 21
img_size = 224
# ... (your existing code)
path = "/Users/euntaeklee/torch_env/torch_class/data/VOC_dataset/"
# Load test data
test_img, test_gt = load_semantic_seg_data(path + 'test/test_img/', path + 'test/test_gt/', path + 'test/test_ce/', img_size=img_size)

# Load the saved model
model_path = '/Users/euntaeklee/Downloads/psp_model/model_10000.pt'
loaded_model = FCRN_8S(num_class)
loaded_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
loaded_model.eval()

# Initialize variables for confusion matrix
num_classes = 21
conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

# Loop through the test dataset
for i in range(len(test_img)):
    img_temp = test_img[i:i + 1, :, :, :].astype(np.float32)
    img_temp = (img_temp / 255.0) * 2 - 1
    img_temp = np.transpose(img_temp, (0, 3, 1, 2))

    with torch.no_grad():
        pred = loaded_model(torch.from_numpy(img_temp.astype(np.float32)).to(Device))

    pred = pred.cpu().numpy()
    pred = np.argmax(pred[0, :, :, :], axis=0)
    gt = test_gt[i, :, :, 0]

    # Flatten the arrays for confusion matrix
    flat_pred = pred.flatten()
    flat_gt = gt.flatten()

    # Update confusion matrix
    conf_matrix += confusion_matrix(flat_gt, flat_pred, labels=np.arange(num_classes))

# Calculate Mean IOU
iou_per_class = np.diag(conf_matrix) / (conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - np.diag(conf_matrix) + 1e-10)
mean_iou = np.mean(iou_per_class)

print('Mean IOU:', mean_iou)
