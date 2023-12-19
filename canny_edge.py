from function import canny_edge_detection

input_train_folder = "/home/hoya9802/Downloads/VOC_dataset/train/train_img"
output_train_folder = "/home/hoya9802/Downloads/VOC_dataset/train/train_ce"

input_test_folder = "/home/hoya9802/Downloads/VOC_dataset/test/test_img"
output_test_folder = "/home/hoya9802/Downloads/VOC_dataset/test/test_ce"

canny_edge_detection(input_train_folder, output_train_folder)
print('-----'*10)
canny_edge_detection(input_test_folder, output_test_folder)
