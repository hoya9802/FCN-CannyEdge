import torch
import torch.nn as nn

class conv_bn_relu(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, pad=1):
        super(conv_bn_relu, self).__init__()
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=(k_size, k_size), padding=pad),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_bn_relu(x)
        return x

class FCN_8S(nn.Module):
    def __init__(self, num_class):
        super(FCN_8S, self).__init__()
        # self.conv_ch = conv_ch
        # [B, C, H, W]
        self.vgg16_part1 = nn.Sequential( # img: [B, 3, 224, 224]
            conv_bn_relu(3, 64),    # [B, 64, 224, 224]
            conv_bn_relu(64, 64),   # [B, 64, 224, 224]
            nn.MaxPool2d(kernel_size=2, stride=2), # [B, 64, 112, 112]
            conv_bn_relu(64, 128), 
            conv_bn_relu(128, 128),  
            nn.MaxPool2d(kernel_size=2, stride=2), # [B, 128, 56, 56]
            conv_bn_relu(128, 256),  
            conv_bn_relu(256, 256),  
            conv_bn_relu(256, 256), 
            nn.MaxPool2d(kernel_size=2, stride=2), # [B, 256, 28, 28]
        )
        self.vgg16_part2 = nn.Sequential(
            conv_bn_relu(256, 512),  # [B, 512, 28, 28]
            conv_bn_relu(512, 512),  
            conv_bn_relu(512, 512),  
            nn.MaxPool2d(kernel_size=2, stride=2), # [B, 512, 14, 14]
        )

        self.vgg16_part3 = nn.Sequential(
            conv_bn_relu(512, 512),  # [B, 512, 14, 14]
            conv_bn_relu(512, 512),  
            conv_bn_relu(512, 512),  
            nn.MaxPool2d(kernel_size=2, stride=2), # [B, 512, 7, 7]

            conv_bn_relu(512, 4096, k_size=1, pad=0),
            conv_bn_relu(4096, 4096, k_size=1, pad=0),
            nn.Conv2d(4096, num_class, kernel_size=(1, 1))
        )
        
        self.pred_part1 = nn.Conv2d(256, num_class, kernel_size=(1, 1))
        self.pred_part2 = nn.Conv2d(512, num_class, kernel_size=(1, 1))
    
        self.up_p3 = nn.ConvTranspose2d(num_class, num_class, kernel_size=4, stride=2, padding=1)
        self.up_p2 = nn.ConvTranspose2d(num_class, num_class, kernel_size=4, stride=2, padding=1)
        self.up_p1 = nn.ConvTranspose2d(num_class, num_class, kernel_size=16, stride=8, padding=4)
    
    def forward(self, x):
        x = self.vgg16_part1(x)
        x_p1 = self.pred_part1(x)

        x = self.vgg16_part2(x)
        x_p2 = self.pred_part2(x)

        x = self.vgg16_part3(x)
        x = self.up_p3(x) + x_p2
        x = self.up_p2(x) + x_p1
        x = self.up_p1(x)

        return x

