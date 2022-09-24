import os
import torch
from dataset import SET
from network import ConvNet
import tqdm
import numpy as np
import torch.optim as optim
from util import evaluate, AverageMeter
import argparse
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter

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


def getB(board):
    b = np.zeros((8, 8))
    mycolor = 1

    for i in range(8):
        for j in range(8):
            Board[0][i][j] = int(b[i][j] == mycolor)
            Board[1][i][j] = int(b[i][j] == -mycolor)

    return placeable(board)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--exp_name', '-e', type=str, required=True,
                            help="The checkpoints and logs will be save in ./checkpoint/$EXP_NAME")
    args = arg_parser.parse_args()
    save_folder = os.path.join('../experiments', args.exp_name)
    ckpt_folder = os.path.join(save_folder, 'ckpt')

    os.makedirs(ckpt_folder, exist_ok=True)
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
    print('load checkpoint from %s' % (read_path))
    checkpoint = torch.load(read_path)
    model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])

    Board = getB(Board)

    model.eval()
    output = model(torch.from_numpy(Board.astype(np.float32)))
    if torch.cuda.is_available():
        output = output.cpu()

    output = output.detach().numpy().reshape(1, 64)
    while True:
        num = np.argmax(output)
        r = int(num / 8)
        c = num % 8
        if Board[0][2][r][c] == 1.:
            print("r:" + str(r) + "\nc:" + str(c) + "\n")
            break
        else:
            output[0][num] = -1e5
