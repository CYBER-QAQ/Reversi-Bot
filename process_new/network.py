import torch.nn as nn


class Block(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 stride,
                 kernel_size):
        super(Block, self).__init__()
        # self.shortcut = nn.Sequential()
        # if stride != 1:
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.conv1 = nn.Conv2d(in_c, out_c * in_c, kernel_size, stride, int((kernel_size - 1) / 2), bias=False)
        self.bn1 = nn.BatchNorm2d(out_c * in_c)
        self.conv2 = nn.Conv2d(out_c * in_c, out_c, kernel_size, 1, int((kernel_size - 1) / 2), bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        out = self.shortcut(x) + y
        out = self.relu(out)
        return out


class ConvNet(nn.Module):
    def __init__(self, num_class=64):
        super(ConvNet, self).__init__()
        pass
        # ----------TODO------------
        # define a network
        self.in_channel = 16
        self.mid_channels = [16, 32, 64]
        self.mid_kernel_size = [[3, 3, 3], [3, 3, 3], [5, 3, 3]]
        # self.mid_padding = [[1, 2, 1], [1, 1, 1], [2, 1, 1]]
        self.conv0 = nn.Conv2d(
            in_channels=5, out_channels=8, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.conv = nn.Conv2d(
            8, self.in_channel, kernel_size=3, padding=1, bias=False,padding_mode='zeros'
        )
        self.bn = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU()

        self.layer1 = self.res_layer(16, 16, 1, 0)
        self.layer2 = self.res_layer(16, 32, 1, 1)
        self.layer3 = self.res_layer(32, 64, 2, 2)
        self.avg_pool = nn.AvgPool2d(3)
        self.liner = nn.Linear(64, num_class)
        self._init_weights()

        # ----------TODO------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight.data,
                    nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def res_layer(self, in_c, out_c, s, order, times=3):
        layer = []
        stride = s
        for i in range(times):
            layer.append(
                Block(in_c, out_c, stride, self.mid_kernel_size[order][i])
            )
            stride = 1
            in_c = out_c
        return nn.Sequential(*layer)

    def forward(self, x):
        # ----------TODO------------·
        # network forwarding
        x = self.conv0(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(-1, 64)
        x = self.liner(x)

        # ----------TODO------------

        return x


if __name__ == '__main__':
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from dataset import SET

    writer = SummaryWriter(log_dir='../experiments/network_structure')
    net = ConvNet()
    train_dataset = SET()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=False, num_workers=2)

    # Write a CNN graph.

    # Please save a figure/screenshot to '../results' for submission.
    for imgs, labels in train_loader:
        writer.add_graph(net, imgs)
        writer.close()
        break
