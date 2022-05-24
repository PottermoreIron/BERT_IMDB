import torch
import logging
import torch.nn as nn
from tqdm import tqdm

import config
from model import BertSentimentClassifier
from metrics import get_f1, bad_case
from transformers import BertTokenizer


def train_epoch(train_loader, model, optimizer, scheduler, epoch):
    model.train()
    train_losses = 0
    for idx, batch_samples in enumerate(tqdm(train_loader)):
        batch_data, batch_labels = batch_samples
        batch_masks = batch_data.gt(0)  # get padding mask
        loss = model(batch_data, token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)
        train_losses += loss.item()
        model.zero_grad()
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
        # Perform updates using calculated gradients
        optimizer.step()
        scheduler.step()
    train_loss = float(train_losses) / len(train_loader)
    logging.info("Epoch: {}, train loss: {}".format(epoch, train_loss))


def train(train_loader, dev_loader, model, optimizer, scheduler, model_dir):
    """train the model and test model performance"""
    # reload weights from restore_dir if specified
    if model_dir is not None and config.load_before:
        model = BertSentimentClassifier.from_pretrained(model_dir)
        model.to(config.device)
        logging.info("--------Load model from {}--------".format(model_dir))
    best_accuracy = 0.0
    patience_counter = 0
    # start training
    for epoch in range(1, config.epoch_num + 1):
        train_epoch(train_loader, model, optimizer, scheduler, epoch)
        val_metrics = evaluate(dev_loader, model, mode='dev')
        accuracy = val_metrics['f1']
        logging.info("Epoch: {}, dev loss: {}, f1 score: {}".format(epoch, val_metrics['loss'], accuracy))
        improve_accuracy = accuracy - best_accuracy
        if improve_accuracy > 1e-5:
            best_accuracy = accuracy
            model.save_pretrained(model_dir)
            logging.info("--------Save best model!--------")
            if improve_accuracy < config.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1
        # Early stopping and logging best f1
        if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch_num:
            logging.info("Best val f1: {}".format(best_accuracy))
            break
    logging.info("Training Finished!")


def evaluate(dev_loader, model, mode='dev'):
    # set model to evaluation mode
    model.eval()
    if mode == 'test':
        bert_tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_dir, do_lower_case=True, skip_special_tokens=True)
    true_tags = []
    pred_tags = []
    sent_data = []
    dev_losses = 0

    with torch.no_grad():
        for idx, batch_samples in enumerate(dev_loader):
            batch_data, batch_tags = batch_samples
            # if mode == 'test':
            #     sent_data.extend([[tokenizer.convert_ids_to_tokens(idx.item()) for idx in indices
            #                        if (idx.item() > 0 and idx.item() != 101)] for indices in batch_data])
            batch_masks = batch_data.gt(0)  # get padding mask, gt(x): get index greater than x
            # compute model output and loss
            loss = model(batch_data,
                         token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)
            dev_losses += loss.item()
            # (batch_size, max_len, num_labels)
            batch_output = model(batch_data, token_type_ids=None, attention_mask=batch_masks)
            # (batch_size, max_len - padding_label_len)
            # batch_output =
            # (batch_size, max_len)
            # batch_tags = batch_tags.to('cpu').numpy()
            pred_tags.extend([indices.max(-1)[1] for indices in batch_output])
            # # (batch_size, max_len - padding_label_len)
            true_tags.extend([idx for indices in batch_tags for idx in indices])        

    # assert len(pred_tags) == len(true_tags)
    # if mode == 'test':
    #     assert len(sent_data) == len(true_tags)

    # logging loss, f1 and report
    print('pr', pred_tags)
    print('tr', true_tags)
    metrics = {'f1': get_f1(true_tags, pred_tags, mode), 'loss': float(dev_losses) / len(dev_loader)}
    return metrics


if __name__ == "__main__":
    pass
