import logging
import numpy as np
from sklearn.model_selection import train_test_split


def dev_split(dataset_dir, dev_split_size=0.1):
    data = np.load(dataset_dir, allow_pickle=True)
    reviews = [review.split() for review in data['review']]
    labels = list(data["label"])
    x_train, x_dev, y_train, y_dev = train_test_split(reviews, labels, test_size=dev_split_size, random_state=0)
    return x_train, x_dev, y_train, y_dev


def loadData(mode, dataset_dir):
    if mode == "train":
        review_train, review_dev, label_train, label_dev = dev_split(dataset_dir)
        return review_train, review_dev, label_train, label_dev
    elif mode == "test":
        data = np.load(dataset_dir, allow_pickle=True)
        review_test = data['review']
        return review_test


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
