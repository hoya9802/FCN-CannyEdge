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


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, bdown=False):
        super(ResBlock, self).__init__()
        stride = 2 if bdown else 1

        self.c_in = c_in
        self.c_out = c_out

        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=(3, 3), padding=1, stride=(stride, stride))
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(c_out)

        if c_in != c_out:
            self.identity = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(stride, stride))

    def forward(self, x):
        y = x  # [8, 512, 64, 64]
        x = self.conv1(x)  # [8, 512, 32, 32]
        x = self.bn1(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.c_in != self.c_out:
            y = self.identity(y)  # [8, 512, 32, 32]

        x = x + y

        x = self.ReLU(x)
        return x


class FCN_8S(nn.Module):
    def __init__(self, num_class):
        super(FCN_8S, self).__init__()
        # self.conv_ch = conv_ch
        # [B, C, H, W]
        self.vgg16_part1 = nn.Sequential(  # img: [B, 6, 224, 224]
            conv_bn_relu(3, 64),  # [B, 64, 224, 224]
            conv_bn_relu(64, 64),  # [B, 64, 224, 224]
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 64, 112, 112]
            conv_bn_relu(64, 128),
            conv_bn_relu(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 128, 56, 56]
            conv_bn_relu(128, 256),
            conv_bn_relu(256, 256),
            conv_bn_relu(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 256, 28, 28]
        )
        self.vgg16_part2 = nn.Sequential(
            conv_bn_relu(256, 512),  # [B, 512, 28, 28]
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 512, 14, 14]
        )

        self.vgg16_part3 = nn.Sequential(
            conv_bn_relu(512, 512),  # [B, 512, 14, 14]
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 512, 7, 7]

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


class FCRN_8S(nn.Module):
    def __init__(self, num_class):
        super(FCRN_8S, self).__init__()
        # self.conv_ch = conv_ch
        # [B, C, H, W]
        self.conv = nn.Conv2d(3, 64, kernel_size=(7, 7), padding=3, stride=(2, 2))  # [112, 112]
        self.bn = nn.BatchNorm2d(64)
        self.ReLU = nn.ReLU()
        # self.maxp = nn.MaxPool2d(kernel_size=2, stride=2) # [56, 56]

        self.resnet34_part1 = nn.Sequential(  # img: [B, 6, 112, 112]
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [56, 56]
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [28, 28]     should be # # [B, 256, 28, 28]
        )

        self.resnet34_part2 = nn.Sequential(
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [14, 14]    # should be # [B, 512, 14, 14]
        )

        self.resnet34_part3 = nn.Sequential(
            ResBlock(512, 512),
            ResBlock(512, 512),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, num_class, kernel_size=(1, 1))  # [7, 7]    # should be # [B, 512, 7, 7]
        )

        self.pred_part1 = nn.Conv2d(256, num_class, kernel_size=(1, 1))
        self.pred_part2 = nn.Conv2d(512, num_class, kernel_size=(1, 1))

        self.up_p3 = nn.ConvTranspose2d(num_class, num_class, kernel_size=4, stride=2, padding=1)
        self.up_p2 = nn.ConvTranspose2d(num_class, num_class, kernel_size=4, stride=2, padding=1)
        self.up_p1 = nn.ConvTranspose2d(num_class, num_class, kernel_size=16, stride=8, padding=4)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.ReLU(x)

        x = self.resnet34_part1(x)
        x_p1 = self.pred_part1(x)

        x = self.resnet34_part2(x)
        x_p2 = self.pred_part2(x)

        x = self.resnet34_part3(x)
        x = self.up_p3(x) + x_p2
        x = self.up_p2(x) + x_p1
        x = self.up_p1(x)

        return x


class DFCRN_8S(nn.Module):
    def __init__(self, num_class):
        super(DFCRN_8S, self).__init__()
        # self.conv_ch = conv_ch
        # [B, C, H, W]
        self.conv = nn.Conv2d(3, 64, kernel_size=(7, 7), padding=3, stride=(2, 2))  # [112, 112]
        self.bn = nn.BatchNorm2d(64)
        self.ReLU = nn.ReLU()
        self.maxp = nn.MaxPool2d(kernel_size=2, stride=2)  # [56, 56]

        self.resnet34_part1 = nn.Sequential(  # img: [B, 6, 112, 112]
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [56, 56]
        )

        self.resnet34_part2 = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [28, 28]     should be # # [B, 256, 28, 28]
        )

        self.resnet34_part3 = nn.Sequential(
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [14, 14]    # should be # [B, 512, 14, 14]
        )

        self.resnet34_part4 = nn.Sequential(
            ResBlock(512, 512),
            ResBlock(512, 512),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, num_class, kernel_size=(1, 1))  # [7, 7]    # should be # [B, 512, 7, 7]
        )

        self.pred_part1 = nn.Conv2d(128, num_class, kernel_size=(1, 1))
        self.pred_part2 = nn.Conv2d(256, num_class, kernel_size=(1, 1))
        self.pred_part3 = nn.Conv2d(512, num_class, kernel_size=(1, 1))

        self.up_p3 = nn.ConvTranspose2d(num_class, num_class, kernel_size=4, stride=2, padding=1)
        self.up_p2 = nn.ConvTranspose2d(num_class, num_class, kernel_size=4, stride=2, padding=1)
        self.up_p1 = nn.ConvTranspose2d(num_class, num_class, kernel_size=16, stride=8, padding=4)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.ReLU(x)

        x = self.resnet34_part1(x)
        x_p1 = self.pred_part1(x)
        x_p1_14 = self.maxp(self.maxp(x_p1))
        x_p1_28 = self.maxp(x_p1)

        x = self.resnet34_part2(x)
        x_p2 = self.pred_part2(x)
        x_p2_14 = self.maxp(x_p2)

        x = self.resnet34_part3(x)
        x_p3 = self.pred_part3(x)

        x = self.resnet34_part4(x)
        x = self.up_p3(x) + x_p3 + x_p2_14 + x_p1_14
        x = self.up_p2(x) + x_p2 + x_p1_28
        x = self.up_p1(x)

        return x