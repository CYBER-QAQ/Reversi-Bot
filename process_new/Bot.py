import os
import torch
import json
import numpy as np
import torch.nn as nn


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class Block(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 stride,
                 kernel_size):
        super(Block, self).__init__()
        self.shortcut = nn.Sequential()
        if stride != 1:
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
        self.mid_kernel_size = [[3, 5, 3], [3, 3, 3], [5, 3, 3]]
        # self.mid_padding = [[1, 2, 1], [1, 1, 1], [2, 1, 1]]
        self.conv = nn.Conv2d(
            3, self.in_channel, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU()

        self.layer1 = self.res_layer(16, 16, 1, 0)
        self.layer2 = self.res_layer(16, 32, 2, 1)
        self.layer3 = self.res_layer(32, 64, 2, 2)

        self.avg_pool = nn.AvgPool2d(3, padding=1)
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


Board = np.zeros((3, 8, 8))
DIR = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))  # 方向向量


def placeable(board):
    avi = np.zeros((8, 8))
    for r in range(8):
        for c in range(8):
            if board[0][r][c] == 0. and board[1][r][c] == 0.:
                for d in range(8):
                    r1 = r + DIR[d][0]
                    c1 = c + DIR[d][1]
                    if not (0 <= r1 < 8 and 0 <= c1 < 8):
                        continue
                    if board[1][r1][c1] == 1.:
                        while 0 <= r1 < 8 and 0 <= c1 < 8 and board[1][r1][c1] == 1.:
                            r1 += DIR[d][0]
                            c1 += DIR[d][1]
                        if 0 <= r1 < 8 and 0 <= c1 < 8 and board[0][r1][c1] == 1.:
                            avi[r][c] = 1.
    board[2] = avi
    return board.reshape(1, 3, 8, 8)


def getB(board, b, mycolor):
    # b = np.zeros((8, 8))
    # mycolor = 1

    for i in range(8):
        for j in range(8):
            board[0][i][j] = float(b[i][j] == mycolor)
            board[1][i][j] = float(b[i][j] == -mycolor)

    return placeable(board)


# 放置棋子，计算新局面
def place(board, x, y, color):
    if x < 0:
        return False
    board[x][y] = color
    valid = False
    for d in range(8):
        i = x + DIR[d][0]
        j = y + DIR[d][1]
        while 0 <= i and i < 8 and 0 <= j and j < 8 and board[i][j] == -color:
            i += DIR[d][0]
            j += DIR[d][1]
        if 0 <= i and i < 8 and 0 <= j and j < 8 and board[i][j] == color:
            while True:
                i -= DIR[d][0]
                j -= DIR[d][1]
                if i == x and j == y:
                    break
                valid = True
                board[i][j] = color
    return valid


# 处理输入，还原棋盘
def initBoard():
    fullInput = json.loads(input())
    requests = fullInput["requests"]
    responses = fullInput["responses"]
    board = np.zeros((8, 8), dtype=np.int)
    board[3][4] = board[4][3] = 1
    board[3][3] = board[4][4] = -1
    myColor = 1
    if requests[0]["x"] >= 0:
        myColor = -1
        place(board, requests[0]["x"], requests[0]["y"], -myColor)
    turn = len(responses)
    for i in range(turn):
        place(board, responses[i]["x"], responses[i]["y"], myColor)
        place(board, requests[i + 1]["x"], requests[i + 1]["y"], -myColor)
    return board, myColor


if __name__ == '__main__':
    # arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument('--exp_name', '-e', type=str, required=True,
    #                         help="The checkpoints and logs will be save in ./checkpoint/$EXP_NAME")
    # args = arg_parser.parse_args()
    # save_folder = os.path.join('./experiments', args.exp_name)
    # ckpt_folder = os.path.join(save_folder, 'ckpt')
    ckpt_folder = os.path.join('./data/experiments', 'ckpt')

    # os.makedirs(ckpt_folder, exist_ok=True)
    # define network
    model = ConvNet()
    if torch.cuda.is_available():
        model = model.cuda()

    # define optimizer
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # define loss
    # criterion = torch.nn.CrossEntropyLoss()

    # load latest checkpoint
    ckpt_lst = os.listdir(ckpt_folder)
    ckpt_lst.sort(key=lambda x: int(x.split('_')[-1]))
    read_path = os.path.join(ckpt_folder, ckpt_lst[-1])
    # print('load checkpoint from %s' % (read_path))
    checkpoint = torch.load(read_path)
    model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])

    b, myColor = initBoard()
    Board = getB(Board, b, myColor)

    model.eval()
    output = model(torch.from_numpy(Board.astype(np.float32)))
    if torch.cuda.is_available():
        output = output.cpu()

    output = output.detach().numpy().reshape(1, 64)

    if np.abs(b).sum() == 5.:
        flag = (b[3][3] == myColor)
        if flag:
            if b[5][4] == -myColor:
                print(json.dumps({"response": {"x": 3, "y": 5}}))
            else:
                print(json.dumps({"response": {"x": 5, "y": 3}}))
        else:
            if b[2][3] == -myColor:
                print(json.dumps({"response": {"x": 4, "y": 2}}))
            else:
                print(json.dumps({"response": {"x": 2, "y": 4}}))


    else:
        while True:
            num = np.argmax(output)
            if output[0][num] == -1e5:
                r = int(-1)
                c = int(-1)
                print(json.dumps({"response": {"x": r, "y": c}}))
                break
            else:
                r = int(num / 8)
                c = int(num % 8)
            if Board[0][2][r][c] == 1.:
                # print("r:" + str(r) + "\nc:" + str(c) + "\n")
                print(json.dumps({"response": {"x": r, "y": c}}))
                break
            else:
                output[0][num] = -1e5
