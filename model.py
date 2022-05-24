from transformers.models.bert.modeling_bert import *


class BertSentimentClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSentimentClassifier, self).__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        self.init_weights()

    def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None, position_ids=None,
                inputs_embeds=None, head_mask=None):
        output = self.bert(input_data,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids,
                           position_ids=position_ids,
                           head_mask=head_mask,
                           inputs_embeds=inputs_embeds,
                           output_hidden_states=True)
        pooled_output = self.dropout(output.pooler_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fnt = CrossEntropyLoss()
            loss = loss_fnt(logits.view(-1, self.num_labels), labels.view(-1).long())
            return loss
        return logits
