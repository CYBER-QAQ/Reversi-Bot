import json
import numpy as np
import random
import pickle
import matplotlib.pyplot as plot
from typing import List

DIR = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))  # 方向向量
Data = np.empty((0, 3, 8, 8))  # 最后一维：我方、对方、可下
Label = []
# init = np.zeros((2, 8, 8), float)  # 初始棋盘： 黑、白
# init[0][3][3] = 1.
# init[0][4][4] = 1.
# init[1][3][4] = 1.
# init[1][4][3] = 1.

cnt = 0


def update(board, x, y, flag):  # flag 0 为我方更新， 1 为对方
    for d in range(8):
        i = x + DIR[d][0]
        j = y + DIR[d][1]
        while 0 <= i < 8 and 0 <= j < 8 and board[int(not flag)][i][j] == 1.:
            i += DIR[d][0]
            j += DIR[d][1]
        if 0 <= i < 8 and 0 <= j < 8 and board[int(flag)][i][j] == 1.:
            while True:
                i -= DIR[d][0]
                j -= DIR[d][1]
                if i == x and j == y:
                    break
                board[int(not flag)][i][j] = 0.
                board[int(flag)][i][j] = 1.


def process(interval):
    global Data
    global Label
    global cnt
    # color = fr.readline()
    color = "white\n"
    if color != "white\n" and color != "black\n":
        print("n1" + color)
        return
    # info = fr.readline()
    info = " f5 d6 c4 d3 c3 f4 c5 b4 b3 c2 e3 e6 b6 c6 a5 f3 b5 e2 e7 g6 c7 a4 a3 f6 c1 f7 f2 f1 g4 h3 g5 d7 c8 a6 e1 g3 h5 h6 h4 d1 a7 d2 g1 d8 b7 b1 a1 a2 b2 a8 g7 b8 h7 g2 h2 h8 g8 h1 e8 f8"
    interval = 3
    # print("going:" + str(float(cnt) / size))
    cnt = cnt + 1
    MeWhite = (color == "white\n")
    board = np.zeros((3, 8, 8))  # 我方、对方、可下
    if MeWhite:
        board[0][3][3] = 1.
        board[0][4][4] = 1.
        board[1][3][4] = 1.
        board[1][4][3] = 1.
    else:
        board[1][3][3] = 1.
        board[1][4][4] = 1.
        board[0][3][4] = 1.
        board[0][4][3] = 1.
    order = 1

    if not MeWhite:
        (r, c) = get_pos(info, order)
        board[0][(r, c)] = 1.
        update(board, r, c, False)
        order += interval
    if order < info.__len__():
        (r, c) = get_pos(info, order)
        board[1][(r, c)] = 1.
        update(board, r, c, True)
        order += interval

    while order < info.__len__():
        r, c = get_pos(info, order)
        if placeable(board, 0, r, c):
            Data = np.vstack((Data, placeable(board)))
            order += interval
            Label.append(r * 8 + c)
            board[0][r][c] = 1.
            update(board, r, c, False)

        if order < info.__len__():
            r, c = get_pos(info, order)
            if placeable(board, 1, r, c):
                board[1][(r, c)] = 1.
                update(board, r, c, True)
                order += interval
    # for i in range(20):
    #     temp = np.zeros((8, 8))
    #     for j in range(8):
    #         for k in range(8):
    #             for l in range(3):
    #                 if Data[i][l][j][k] == 1.:
    #                     temp[j][k] = l + 1
    #     print(str(i) + ":\n")
    #     print(temp)
    # print()


def get_pos(info, pos):
    return tuple((ord(info[pos]) - ord('a'), int(info[pos + 1]) - 1))


def placeable(board, Me=-1, R=-1., C=-1.):  # Me：即将落子的颜色， 0 为我，1为你
    if Me == -1 and R == -1. and C == -1.:
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
    elif 0 <= R < 8 and 0 <= C < 8:
        if Me == 0 or Me == 1:
            you = (Me + 1) % 2
            if board[0][R][C] == 0. and board[1][R][C] == 0.:
                for d in range(8):
                    r1 = R + DIR[d][0]
                    c1 = C + DIR[d][1]
                    if not (0 <= r1 < 8 and 0 <= c1 < 8):
                        continue
                    if board[you][r1][c1] == 1.:
                        while 0 <= r1 < 8 and 0 <= c1 < 8 and board[you][r1][c1] == 1.:
                            r1 += DIR[d][0]
                            c1 += DIR[d][1]
                        if 0 <= r1 < 8 and 0 <= c1 < 8 and board[Me][r1][c1] == 1.:
                            return True
                return False
            else:
                print("n6\n")
                return None
        else:
            print("n5\n")
            return None
    else:
        print("n5\n")
        return None


# 放置棋子，计算新局面
def place(board, x, y, color):
    if x < 0:
        return False
    board[x][y] = color
    valid = False
    for d in range(8):
        i = x + DIR[d][0]
        j = y + DIR[d][1]
        while 0 <= i < 8 and 0 <= j < 8 and board[i][j] == -color:
            i += DIR[d][0]
            j += DIR[d][1]
        if 0 <= i < 8 and 0 <= j < 8 and board[i][j] == color:
            while True:
                i -= DIR[d][0]
                j -= DIR[d][1]
                if i == x and j == y:
                    break
                valid = True
                board[i][j] = color
    return valid


# 随机产生决策
def randplace(board, color):
    x = y = -1
    moves = []
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0:
                newBoard = board.copy()
                if place(newBoard, i, j, color):
                    moves.append((i, j))
    if len(moves) == 0:
        return -1, -1
    return random.choice(moves)


def simple_count(f_name):
    global lines
    for _ in open(f_name):
        lines += 1
    return lines


process(3)
#
#
# file_name = "../data/100-3-4.txt"
# # file_name = "../data/20-0-3.txt"
# lines = 0
# simple_count(file_name)
# print(lines)
# fr = open(file_name, "r")
# size = int(lines / 2 - 1)  # 根据文件行数确定
# #
# # data_list = ["./SET1/data1-1", "./SET1/data1-2", "./SET1/data1-3", "./SET1/data1-4", "./SET1/data1-5"]
# # label_list = ["./SET1/label1-1", "./SET1/label1-2", "./SET1/label1-3", "./SET1/label1-4", "./SET1/label1-5"]
# #
#
# # data_list = ["./SET1/data2-1", "./SET1/data2-2", "./SET1/data2-3", "./SET1/data2-4", "./SET1/data2-5"]
# # label_list = ["./SET1/label2-1", "./SET1/label2-2", "./SET1/label2-3", "./SET1/label2-4", "./SET1/label2-5"]
# #
# # data_list = ["./SET1/data3-1", "./SET1/data3-2", "./SET1/data3-3", "./SET1/data3-4", "./SET1/data3-5"]
# # label_list = ["./SET1/label3-1", "./SET1/label3-2", "./SET1/label3-3", "./SET1/label3-4", "./SET1/label3-5"]
# # for j in range(5):
# #
# #     for i in range(int(size / 5)):
# #         process(5)
# #     with open(data_list[j], "wb") as wd:
# #         pickle.dump(Data, wd)
# #     with open(label_list[j], "wb") as wl:
# #         pickle.dump(Label, wl)
# #
# #     Data = np.empty((0, 3, 8, 8))  # 最后一维：我方、对方、可下
# #     Label = []
# #
# #     with open(data_list[j], "rb") as rd:
# #         D = pickle.load(rd)
# #     with open(label_list[j], "rb") as rl:
# #         L = pickle.load(rl)
# #     print(D.shape)
# #     print(len(L))
# #
# #     D = np.empty((0, 3, 8, 8))  # 最后一维：我方、对方、可下
# #     L = []
#
# for i in range(size):
#     process(5)
# with open("./SET1/test_data1", "wb") as wd:
#     pickle.dump(Data, wd)
# with open("./SET1/test_label1", "wb") as wl:
#     pickle.dump(Label, wl)
# with open("./SET1/test_data1", "rb") as rd:
#     D = pickle.load(rd)
# with open("./SET1/test_label1", "rb") as rl:
#     L = pickle.load(rl)
# print(D.shape)
# print(len(L))
# D = np.empty((0, 3, 8, 8))  # 最后一维：我方、对方、可下
# L = []
