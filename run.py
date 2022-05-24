import config
import logging
import utils
import numpy as np
import pandas as pd
import torch
from data_process import IMDBDataProcessor
from data_loader import IMDBDataset
from utils import loadData
from torch.utils.data import DataLoader
from model import BertSentimentClassifier
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW
from train import train

utils.set_logger(config.log_dir)
logging.info("device: {}".format(config.device))


# processor = IMDBDataProcessor(config.data_dir)
# processor.process()
# logging.info("--------Process Done!--------")
# review_train, review_dev, label_train, label_dev = loadData('train', config.train_data_dir2)
# train_dataset = IMDBDataset(review_train, label_train, config.pretrained_model_dir, config.device)
# dev_dataset = IMDBDataset(review_dev, label_dev, config.pretrained_model_dir, config.device)
# logging.info("--------Dataset Build!--------")
# train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
# dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=dev_dataset.collate_fn)
# logging.info("--------Get Dataloader!--------")
# train_size = len(train_dataset)
#
# device = config.device
# model = BertSentimentClassifier.from_pretrained(config.pretrained_model_dir, num_labels=config.num_labels)
# model.to(device)
#
#
# if config.full_fine_tuning:
#     # model.named_parameters(): [bert, bilstm, classifier, crf]
#     bert_optimizer = list(model.bert.named_parameters())
#     classifier_optimizer = list(model.classifier.named_parameters())
#     no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
#          'weight_decay': config.weight_decay},
#         {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
#          'weight_decay': 0.0},
#         {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
#          'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
#         {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
#          'lr': config.learning_rate * 5, 'weight_decay': 0.0},
#     ]
# # only fine-tune the head classifier
# else:
#     param_optimizer = list(model.classifier.named_parameters())
#     optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
# optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, correct_bias=False,no_deprecation_warning=True)
# train_steps_per_epoch = train_size // config.batch_size
# scheduler = get_cosine_schedule_with_warmup(optimizer,
#                                             num_warmup_steps=(config.epoch_num // 10) * train_steps_per_epoch,
#                                             num_training_steps=config.epoch_num * train_steps_per_epoch)
# logging.info("--------Start Training!--------")
# train(train_loader, dev_loader, model, optimizer, scheduler, config.save_model_dir)


def predict():
    data = np.load(config.test_data_dir2, allow_pickle=True)
    review_test = [review.split() for review in data['review']]
    label_test = [1 for _ in range(len(review_test))]
    test_dataset = IMDBDataset(review_test, label_test, config.pretrained_model_dir, config.device)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, collate_fn=test_dataset.collate_fn)
    model = BertSentimentClassifier.from_pretrained(config.save_model_dir)
    model.to(config.device)
    model.eval()
    pred_tags = []
    with torch.no_grad():
        for idx, batch_samples in enumerate(test_loader):
            batch_data, batch_tags = batch_samples
            batch_masks = batch_data.gt(0)
            batch_output = model(batch_data, token_type_ids=None, attention_mask=batch_masks)
            pred_tags.extend([indices.max(-1)[1] for indices in batch_output])
    print(pred_tags)


if __name__ == "__main__":
    # predict()
    label_data = np.load('test.npy').tolist()
    id_data = [idx[0] for idx in pd.read_csv('data/test.tsv', sep='\t', header=0, usecols=['id']).values.tolist()]
    dataframe = pd.DataFrame({'id': id_data, 'sentiment': label_data})
    dataframe.to_csv("Bert.csv", index=False, sep=',')
