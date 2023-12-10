import numpy as np
from function import *
from network import *
import time
import matplotlib.pyplot as plt

Device = torch.device("mps")
print(Device)

num_class = 21
batch_size = 16
img_size = 224

model_save_path = '/Users/euntaeklee/torch_env/torch_class/FCN_CE/model/'
image_save_path = '/Users/euntaeklee/torch_env/torch_class/FCN_CE/seg_result/'
path = "/Users/euntaeklee/torch_env/torch_class/data/VOC_dataset/"

print('load_image....')
train_img, train_gt = load_semantic_seg_data(path + 'train/train_img/', path + 'train/train_gt/', path + 'train/train_ce/', img_size=img_size)
test_img, test_gt = load_semantic_seg_data(path + 'test/test_img/', path + 'test/test_gt/', path + 'test/test_ce/', img_size=img_size)


print('load_image_finish')

train_loss_history = []

model = FCN_8S(num_class).to(Device)


learning_rate = 0.01
num_iter = 300000
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=5e-4, momentum=0.9)

start_time = time.time()
for it in range(num_iter):
    if it >= 100000 and it < 200000:
        optimizer.param_groups[0]['lr'] = 0.001
    if it >= 200000:
        optimizer.param_groups[0]['lr'] = 0.0001

    # batch_img = [B, 224, 224, 3] / batch_gt = [B, 224, 224, 1]
    batch_img, batch_gt = Mini_batch_training_seg(train_img, train_gt, batch_size, img_size)
    batch_img = np.transpose(batch_img, (0, 3, 1, 2))

    # ---- training step
    model.train()
    optimizer.zero_grad()
    pred = model(torch.from_numpy(batch_img.astype(np.float32)).to(Device)) # [batch, 10] = [64, 10]
    
    gt_tensor = torch.tensor(batch_gt, dtype=torch.long).to(Device)
    gt_tensor = torch.nn.functional.one_hot(gt_tensor, num_class).squeeze()
    gt_tensor = torch.permute(gt_tensor, (0, 3, 1, 2)) # np.transpose랑 같다.

    # 만약 pytorch version이 낮아서 torch.permute을 one_hot()이후 지원하지 않으면 아래 코드를 위 2개 코드 대신 사용!
    # gt_tensor = gt_tensor[:,:,:,0]

    train_loss = torch.nn.functional.cross_entropy(pred, gt_tensor.type(torch.float32))
    train_loss.backward()
    optimizer.step()

    if it % 100 == 0:
        consum_time = time.time() - start_time
        train_loss_history.append(train_loss.item())
        print('iter: %d   train loss: %.5f   lr: %.5f   time: %.4f'
              %(it, train_loss.item(), optimizer.param_groups[0]['lr'], consum_time))
        model.eval()
        start_time = time.time()

    if it % 10000 == 0: # and it != 0
        print('SAVING MODEL')
        if not os.path.isdir(model_save_path):
            os.makedirs(model_save_path)

        torch.save(model.state_dict(), model_save_path + 'model_%d.pt' % it)
        print('SAVING MODEL FINISH')

        for itest in range(100):
            img_temp = test_img[itest:itest + 1, :, :, :].astype(np.float32)
            img_temp = (img_temp / 255.0) * 2 - 1  # [1, 28, 28]
            img_temp = np.transpose(img_temp, (0, 3, 1, 2))

            with torch.no_grad():
                pred = model(torch.from_numpy(img_temp.astype(np.float32)).to(Device))

            pred = pred.cpu().numpy()
            pred = np.argmax(pred[0, :, :, :], axis=0)
            pred = pred[:, :, np.newaxis]

            test_save = np.zeros(shape=(img_size, img_size, 3), dtype=np.uint8)
            for ic in range(len(VOC_COLORMAP)):
                code = VOC_COLORMAP[ic]
                test_save[np.where(np.all(pred == ic, axis=-1))] = code

            big_paper = np.ones(shape=(img_size, 2 * img_size, 3), dtype=np.uint8)
            big_paper[:, :img_size, :] = test_img[itest:itest + 1, :, :, :]
            big_paper[:, img_size:, :] = test_save

            temp = image_save_path + '%d/' %it
            if not os.path.isdir(temp):
                os.makedirs(temp)

            cv2.imwrite(temp + '%d.png' %(itest), big_paper)
        


plt.figure(figsize=(12, 5))
plt.plot(train_loss_history)
plt.title('Training Set Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.savefig('FCN_CE/train_loss.png')
plt.show()
