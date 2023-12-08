from function import canny_edge_detection

input_train_folder = "/Users/euntaeklee/torch_env/torch_class/data/VOC_dataset/train/train_img"
output_train_folder = "/Users/euntaeklee/torch_env/torch_class/data/VOC_dataset/train/train_ce"

input_test_folder = "/Users/euntaeklee/torch_env/torch_class/data/VOC_dataset/test/test_img"
output_test_folder = "/Users/euntaeklee/torch_env/torch_class/data/VOC_dataset/test/test_ce"

canny_edge_detection(input_train_folder, output_train_folder)
