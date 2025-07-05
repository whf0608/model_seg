from torch import nn


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv_layers):
        super(_DecoderBlock, self).__init__()
        middle_channels = in_channels / 2
        layers = [
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        ]
        layers += [
                      nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
                      nn.BatchNorm2d(middle_channels),
                      nn.ReLU(inplace=True),
                  ] * (num_conv_layers - 2)
        layers += [
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


class DecoderBottleneck(nn.Module):
    def __init__(self, inchannels):
        super(DecoderBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, inchannels // 4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inchannels // 4)
        self.conv2 = nn.ConvTranspose2d(inchannels // 4, inchannels // 4, kernel_size=2, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(inchannels // 4)
        self.conv3 = nn.Conv2d(inchannels // 4, inchannels // 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(inchannels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.ConvTranspose2d(inchannels, inchannels // 2, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(inchannels // 2))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class LastBottleneck(nn.Module):
    def __init__(self, inchannels):
        super(LastBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, inchannels // 4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inchannels // 4)
        self.conv2 = nn.Conv2d(inchannels // 4, inchannels // 4, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inchannels // 4)
        self.conv3 = nn.Conv2d(inchannels // 4, inchannels // 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(inchannels // 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(inchannels, inchannels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(inchannels // 4))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out