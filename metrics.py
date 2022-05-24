import torch
import config
import os
import logging
from sklearn.metrics import f1_score


def get_f1(label_true, label_pred, mode='dev'):
    return f1_score([tr.cpu().numpy() for tr in label_true], [pr.cpu().numpy() for pr in label_pred], average="micro")


def bad_case(label_true, label_pred, data):
    if not os.path.exists(config.case_dir):
        os.system(r"touch {}".format(config.case_dir))  # 调用系统命令行来创建文件
    output = open(config.case_dir, 'w')
    for idx, (t, p) in enumerate(zip(y_true, y_pred)):
        if t == p:
            continue
        else:
            output.write("bad case " + str(idx) + ": \n")
            output.write("sentence: " + str(data[idx]) + "\n")
            output.write("golden label: " + str(t) + "\n")
            output.write("model pred: " + str(p) + "\n")
    logging.info("--------Bad Cases reserved !--------")


# y_true = torch.Tensor([[1, 1, 0, 0, 1, 1, 1], [1, 1, 0, 0, 1, 1, 1]])
# y_pred = torch.Tensor([[1, 1, 1, 0, 1, 1, 1], [1, 1, 0, 0, 1, 1, 1]])
# get_accuracy(y_true, y_pred)
