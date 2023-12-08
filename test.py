import cv2

def get_image_dimensions(image_path):
    # 이미지 불러오기
    img = cv2.imread(image_path)

    # 이미지 차원 확인
    dimensions = img.shape

    return dimensions

# 이미지 파일 경로 지정
image_path = '/Users/euntaeklee/torch_env/torch_class/data/VOC_dataset/train/train_ce/2007_000039.jpg'

# 이미지 차원 확인
dimensions = get_image_dimensions(image_path)

print(f"Image dimensions: {dimensions}")
print(f"Number of dimensions: {len(dimensions)}")