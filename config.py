import os
import torch

working_dir = os.getcwd()
data_dir = working_dir + '/data/'
train_data_dir = data_dir + 'train.tsv'
train_data_dir2 = data_dir + 'train.npz'
test_data_dir = data_dir + 'test.tsv'
test_data_dir2 = data_dir + 'test.npz'
pretrained_model_dir = data_dir+r'pretrained_model/bert-base-uncased'

save_model_dir = working_dir + '/model'
log_dir = save_model_dir + '/train.log'
case_dir = working_dir + '/case/bad_case.txt'

# 训练集、验证集划分比例
dev_split_size = 0.1

# 是否加载训练好的模型
load_before = False

# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# hyper-parameter
learning_rate = 3e-5
weight_decay = 0.01
clip_grad = 5

batch_size = 16
epoch_num = 50
min_epoch_num = 5
patience = 0.0002
patience_num = 10

num_labels = 2

gpu = '' if not torch.cuda.is_available() else 0
if gpu != '':
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")

