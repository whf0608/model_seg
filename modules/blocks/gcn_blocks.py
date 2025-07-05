from torch import nn
import torchvision

class Block_Resnet_GCN(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, stride=1):
        super(Block_Resnet_GCN, self).__init__()
        self.conv11 = nn.Conv2d(in_channels, out_channels, bias=False, stride=stride,
                                kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0), )
        self.bn11 = nn.BatchNorm2d(out_channels)
        self.relu11 = nn.ReLU(inplace=True)
        self.conv12 = nn.Conv2d(out_channels, out_channels, bias=False, stride=stride,
                                kernel_size=(1, kernel_size), padding=(0, kernel_size // 2))
        self.bn12 = nn.BatchNorm2d(out_channels)
        self.relu12 = nn.ReLU(inplace=True)

        self.conv21 = nn.Conv2d(in_channels, out_channels, bias=False, stride=stride,
                                kernel_size=(1, kernel_size), padding=(0, kernel_size // 2))
        self.bn21 = nn.BatchNorm2d(out_channels)
        self.relu21 = nn.ReLU(inplace=True)
        self.conv22 = nn.Conv2d(out_channels, out_channels, bias=False, stride=stride,
                                kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0))
        self.bn22 = nn.BatchNorm2d(out_channels)
        self.relu22 = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv11(x)
        x1 = self.bn11(x1)
        x1 = self.relu11(x1)
        x1 = self.conv12(x1)
        x1 = self.bn12(x1)
        x1 = self.relu12(x1)

        x2 = self.conv21(x)
        x2 = self.bn21(x2)
        x2 = self.relu21(x2)
        x2 = self.conv22(x2)
        x2 = self.bn22(x2)
        x2 = self.relu22(x2)

        x = x1 + x2
        return x


class BottleneckGCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, out_channels_gcn, stride=1):
        super(BottleneckGCN, self).__init__()
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.downsample = None

        self.gcn = Block_Resnet_GCN(kernel_size, in_channels, out_channels_gcn)
        self.conv1x1 = nn.Conv2d(out_channels_gcn, out_channels, 1, stride=stride, bias=False)
        self.bn1x1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)

        x = self.gcn(x)
        x = self.conv1x1(x)
        x = self.bn1x1(x)

        x += identity
        return x


class ResnetGCN(nn.Module):
    def __init__(self, in_channels, backbone, out_channels_gcn=(85, 128), kernel_sizes=(5, 7)):
        super(ResnetGCN, self).__init__()
        resnet = getattr(torchvision.models, backbone)(pretrained=False)

        if in_channels == 3:
            conv1 = resnet.conv1
        else:
            conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial = nn.Sequential(
            conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool)

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = nn.Sequential(
            BottleneckGCN(512, 1024, kernel_sizes[0], out_channels_gcn[0], stride=2),
            *[BottleneckGCN(1024, 1024, kernel_sizes[0], out_channels_gcn[0])] * 5)
        self.layer4 = nn.Sequential(
            BottleneckGCN(1024, 2048, kernel_sizes[1], out_channels_gcn[1], stride=2),
            *[BottleneckGCN(1024, 1024, kernel_sizes[1], out_channels_gcn[1])] * 5)
        # initialize_weights(self)

    def forward(self, x):
        x = self.initial(x)
        conv1_sz = (x.size(2), x.size(3))
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4, conv1_sz