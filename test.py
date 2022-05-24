import config
from utils import loadData
from data_loader import IMDBDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

review_train, review_dev, label_train, label_dev = loadData('train', config.train_data_dir2)
train_dataset = IMDBDataset(review_train, label_train, config.pretrained_model_dir, config.device)
dev_dataset = IMDBDataset(review_dev, label_dev, config.pretrained_model_dir, config.device)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=train_dataset.collate_fn,
                          shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, collate_fn=train_dataset.collate_fn, shuffle=True)

device = config.device
model = BertSentimentClassifier()
model.to(device)

if config.full_fine_tuning:
    # model.named_parameters(): [bert, bilstm, classifier, crf]
    bert_optimizer = list(model.bert.named_parameters())
    classifier_optimizer = list(model.classifier.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
         'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
        {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
         'lr': config.learning_rate * 5, 'weight_decay': 0.0},
    ]
# only fine-tune the head classifier
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, correct_bias=False,
                  no_deprecation_warning=True)
train_steps_per_epoch = train_size // config.batch_size
scheduler = get_cosine_schedule_with_warmup(optimizer,
                                            num_warmup_steps=(config.epoch_num // 10) * train_steps_per_epoch,
                                            num_training_steps=config.epoch_num * train_steps_per_epoch)
logging.info("--------Start Training!--------")
train(train_loader, dev_loader, model, optimizer, scheduler, config.save_model_dir)