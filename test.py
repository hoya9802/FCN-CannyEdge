import cv2
import numpy as np

def get_image_dimensions(image_path):
    # 이미지 불러오기
    img = cv2.imread(image_path)

    # 이미지 차원 확인
    dimensions = img.shape

    return img, dimensions

# 이미지 파일 경로 지정
ce_path = '/Users/euntaeklee/torch_env/torch_class/data/VOC_dataset/train/train_ce/2007_005702.jpg'
img_path = '/Users/euntaeklee/torch_env/torch_class/data/VOC_dataset/train/train_img/2007_005702.jpg'

# 이미지 차원 확인
ce, ce_dimensions = get_image_dimensions(ce_path)
img, img_dimensions = get_image_dimensions(img_path)

merge = np.maximum(img, ce)

# print(f"Ce dimensions: {ce_dimensions} | Img dimensions: {img_dimensions} | Merge dimensions: {merge.shape}")

cv2.imshow("CE_Image", ce)
cv2.imshow("Img_Image", img)
cv2.imshow('merge_img', merge)


cv2.waitKey()
cv2.destroyAllWindows()