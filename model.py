import torch
from torch import nn
import torch.nn.functional as F



class ConvBlock(nn.Module):
    def __init__(self, input_ch=3, output_ch=64, activf=nn.ReLU, bias=True):
        super().__init__()

        self.conv1 = nn.Conv2d(input_ch, output_ch, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(output_ch, output_ch, 3, 1, 1, bias=bias)
        self.activf = activf

        self.conv_block = nn.Sequential(
            self.conv1,
            self.activf(inplace=True),
            self.conv2,
            self.activf(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


class UpConv(nn.Module):
    def __init__(self, input_ch=64, output_ch=32, bias=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(input_ch, output_ch, 2, 2, bias=bias)
        self.conv_block = nn.Sequential(self.conv)

    def forward(self, x):
        return self.conv_block(x)


class UNetModule(nn.Module):
    def __init__(self, input_ch, output_ch, base_ch):
        super().__init__()

        self.conv1 = ConvBlock(input_ch, base_ch)
        self.conv2 = ConvBlock(base_ch, 2* base_ch)
        self.conv3 = ConvBlock(2 * base_ch, 4 * base_ch)
        self.conv4 = ConvBlock(4 * base_ch, 8 * base_ch)
        self.conv5 = ConvBlock(8 * base_ch, 16 * base_ch)

        self.upconv1 = UpConv(16 * base_ch, 8 * base_ch)
        self.conv6 = ConvBlock(16 * base_ch, 8 * base_ch)
        self.upconv2 = UpConv(8 * base_ch, 4 * base_ch)
        self.conv7 = ConvBlock(8 * base_ch, 4 * base_ch)
        self.upconv3 = UpConv(4 * base_ch, 2 * base_ch)
        self.conv8 = ConvBlock(4 * base_ch, 2 * base_ch)
        self.upconv4 = UpConv(2 * base_ch, base_ch)
        self.conv9 = ConvBlock(2 * base_ch, base_ch)

        self.outconv = nn.Conv2d(base_ch, output_ch, 1, bias=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x = F.max_pool2d(x1, 2, 2)

        x2 = self.conv2(x)
        x = F.max_pool2d(x2, 2, 2)

        x3 = self.conv3(x)
        x = F.max_pool2d(x3, 2, 2)

        x4 = self.conv4(x)
        x = F.max_pool2d(x4, 2, 2)

        x = self.conv5(x)
        x = self.upconv1(x)
        x = torch.cat((x4, x), dim=1)

        x = self.conv6(x)
        x = self.upconv2(x)
        x = torch.cat((x3, x), dim=1)

        x = self.conv7(x)
        x = self.upconv3(x)
        x = torch.cat((x2, x), dim=1)

        x = self.conv8(x)
        x = self.upconv4(x)
        x = torch.cat((x1, x), dim=1)

        x = self.conv9(x)
        x = self.outconv(x)

        return x


class RRWNet(nn.Module):
    def __init__(self, input_ch=3, output_ch=3, base_ch=64, iterations=5):
        super().__init__()
        self.first_u = UNetModule(input_ch, output_ch, base_ch)
        self.second_u = UNetModule(output_ch, 2, base_ch)
        self.iterations = iterations

    def forward(self, x):
        predictions = []

        pred_1 = self.first_u(x)
        predictions.append(pred_1)
        bv_logits = pred_1[:, 2:3, :, :]
        pred_1 = torch.sigmoid(pred_1)
        bv = pred_1[:, 2:3, :, :]

        pred_2 = self.second_u(pred_1)
        predictions.append(torch.cat((pred_2, bv_logits), dim=1))

        for _ in range(self.iterations):
            pred_2 = torch.sigmoid(pred_2)
            pred_2 = torch.cat((pred_2, bv), dim=1)
            pred_2 = self.second_u(pred_2)
            predictions.append(torch.cat((pred_2, bv_logits), dim=1))

        return predictions

    def refine(self, x):
        predictions = []
        bv = x[:, 2:3, :, :]

        pred_2 = self.second_u(x)
        predictions.append(torch.cat((torch.sigmoid(pred_2), bv), dim=1))

        for _ in range(self.iterations):
            pred_2 = torch.sigmoid(pred_2)
            pred_2 = torch.cat((pred_2, bv), dim=1)
            pred_2 = self.second_u(pred_2)
            predictions.append(torch.cat((torch.sigmoid(pred_2), bv), dim=1))

        return predictions
