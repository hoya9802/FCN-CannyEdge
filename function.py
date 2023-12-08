import numpy as np
import os
import cv2

VOC_COLORMAP = [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128], [128, 0, 0], [128, 0, 128],
                [128, 128, 0], [128, 128, 128], [0, 0, 64], [0, 0, 192], [0, 128, 64], [0, 128, 192],
                [128, 0, 64], [128, 0, 192], [128, 128, 64], [128, 128, 192], [0, 64, 0], [0, 64, 128],
                [0, 192, 0], [0, 192, 128], [128, 64, 0]
                ]

def load_semantic_seg_data(img_path, gt_path, img_size):
     # --- load image
    img_names = os.listdir(img_path)
    gt_names = os.listdir(gt_path)

    img_names = sorted(img_names, key=lambda x: int(''.join(filter(str.isdigit, x))))
    gt_names = sorted(gt_names, key=lambda x: int(''.join(filter(str.isdigit, x))))
    print(img_names[:10])
    print(gt_names[:10])
    imgs = np.zeros((len(img_names), img_size, img_size, 3), dtype=np.uint8) # [B, H, W, C]
    gts = np.zeros((len(gt_names), img_size, img_size, 3), dtype=np.uint8) # [B, H, W, C]

    for it in range(len(img_names)):
        print('%d / %d' %(it, len(img_names)))

        img = cv2.imread(img_path + img_names[it])
        img = cv2.resize(img, (img_size, img_size))
        imgs[it, :, :, :] = img

    # ------- gt!!!
    for it in range(len(gt_names)):
        print('%d / %d' %(it, len(gt_names)))
        gt = cv2.imread(gt_path + gt_names[it])
        gt_index = np.zeros(shape=(gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
        for ic in range(len(VOC_COLORMAP)):
            code = VOC_COLORMAP[ic]
            gt_index[np.where(np.all(gt == code, axis=-1))] = ic


        gt_index = cv2.resize(gt_index, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        gts[it, :, :, :] = gt_index


    return imgs, gts


def Mini_batch_training_seg(train_img, train_gt, batch_size, img_size):
    batch_img = np.zeros((batch_size, img_size, img_size, 3))
    batch_gt = np.zeros((batch_size, img_size, img_size, 1))

    # train_img = [200xxx, 128, 128, 3]
    rand_num = np.random.randint(0, train_img.shape[0], size=batch_size) # [2, 19, 77, 10]

    # pixel normalization : 0 - 255 / -1 - 1
    for it in range(batch_size):
        temp = rand_num[it]
        batch_img[it, :, :] = (train_img[temp, :, :, :] / 255.0) * 2 - 1 # (0 ~ 1) x 2 -> 0 ~ 2 -> -1 - 1
        batch_gt[it] = train_gt[temp, :, :, 0:1]

    return batch_img, batch_gt