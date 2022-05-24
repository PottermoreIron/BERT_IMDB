import os
import pandas as pd
import numpy as np
import re
import logging
import config
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


def cleanData(review, remove_stops=False, words_list=False):
    url_pattern = re.compile(
        r'(?:(?:https?|ftp|file):\/\/|www\.|ftp\.)(?:\([-A-Za-z0-9+&@#/%=~_|$?!:,.]*\)|[-A-Za-z0-9+&@#/%=~_|$?!:,.])*(?:\([-A-Za-z0-9+&@#/%=~_|$?!:,.]*\)|[A-Za-z0-9+&@#/%=~_|$])')
    alpha_num_pattern = re.compile(r'[^a-zA-Z]')
    dot_pattern = re.compile(r'\.+')
    review = re.sub(url_pattern, " ", review)
    review = re.sub(dot_pattern, " ", review)
    res_text = re.sub(alpha_num_pattern, " ", BeautifulSoup(review, features="html.parser").get_text()).lower().split()
    if remove_stops:
        stops = set(stopwords.words("english"))
        res_text = [w for w in res_text if w not in stops]
    if words_list:
        return res_text
    return " ".join(res_text)


class IMDBDataProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def process(self):
        for mode in ['train', 'test']:
            self.preProcess(mode)

    def preProcess(self, mode):
        input_dir = self.data_dir + mode + '.tsv'
        output_dir = self.data_dir + mode + '.npz'
        if os.path.exists(output_dir):
            return
        original_data = pd.read_csv(input_dir, header=0, delimiter="\t", quoting=3)
        reviews = []
        labels = []
        for review in original_data['review']:
            reviews.append(cleanData(review, True))
        if mode == 'train':
            for label in original_data['sentiment']:
                labels.append(label)
        np.savez_compressed(output_dir, review=reviews, label=labels)
        logging.info("--------{} data process DONE!--------".format(mode))
