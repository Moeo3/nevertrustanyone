import torch.nn as nn
import torchvision
import torch

class VggPretrained(nn.Module):
    def __init__(self):
        super(VggPretrained, self).__init__()
        features = torchvision.models.vgg19(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_4 = nn.Sequential()
        self.to_relu_4_4 = nn.Sequential()
        self.to_relu_5_4 = nn.Sequential()

        first_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        first_conv.weight.data = torch.mean(features[0].weight.data, dim=1, keepdim=True)
        first_conv.bias.data = features[0].bias.data
        self.to_relu_1_2.add_module('0', first_conv)

        for x in range(1, 4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 18):
            self.to_relu_3_4.add_module(str(x), features[x])
        for x in range(18, 27):
            self.to_relu_4_4.add_module(str(x), features[x])
        for x in range(27, 36):
            self.to_relu_5_4.add_module(str(x), features[x])

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.to_relu_1_2(x)
        h_relu_1_2 = x
        x = self.to_relu_2_2(x)
        h_relu_2_2 = x
        x = self.to_relu_3_4(x)
        h_relu_3_4 = x
        x = self.to_relu_4_4(x)
        h_relu_4_4 = x
        x = self.to_relu_5_4(x)
        h_relu_5_4 = x

        return h_relu_1_2, h_relu_2_2, h_relu_3_4, h_relu_4_4, h_relu_5_4
