import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class IMDBDataset(Dataset):
    def __init__(self, reviews, labels, pretrained_model, device, review_pad_idx=0):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower_case=True)
        self.device = device
        self.dataset = self.preProcess(reviews, labels)
        self.review_pad_idx = review_pad_idx

    def preProcess(self, origin_reviews, origin_labels):
        data = []
        reviews = []
        for review in origin_reviews:
            words = ['[CLS]'] + review
            reviews.append(self.tokenizer.convert_tokens_to_ids(words))
        labels = origin_labels
        for review, label in zip(reviews, labels):
            data.append((review, label))
        return data

    def __getitem__(self, idx):
        """sample data to get batch"""
        review = self.dataset[idx][0]
        label = self.dataset[idx][1]
        return [review, label]

    def __len__(self):
        """get dataset size"""
        return len(self.dataset)

    def collate_fn(self, batch):
        reviews = [x[0] for x in batch]
        labels = [x[1] for x in batch]
        batch_len = len(reviews)
        max_len = max([len(r) for r in reviews])
        if max_len > 512:
            max_len = 512
        batch_data = self.review_pad_idx * np.ones((batch_len, max_len))
        for i in range(batch_len):
            cur_len = len(reviews[i])
            if cur_len <= 512:
                batch_data[i][:cur_len] = reviews[i]
            else:
                batch_data[i] = reviews[i][0:512]
        batch_labels = np.ones((batch_len, 1))
        for i in range(batch_len):
            batch_labels[i] = labels[i]
        batch_data = torch.tensor(batch_data, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.float)
        batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
        return [batch_data, batch_labels]
