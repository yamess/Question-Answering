import torch.nn as nn
from transformers import (
    BertModel,
    AlbertModel
)

class BertForQA(nn.Module):
    def __init__(self, bert_path, dropout):
        super(BertForQA, self).__init__()
        self.qa_bert = BertModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.qa_bert.config.hidden_size, 2)

    def forward(self, input_ids, mask, token_type_ids):
        sequence_output, _ = self.qa_bert(
            input_ids=input_ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        logits = self.dropout(sequence_output)
        logits = self.fc(logits)  # (batch_size, num_tokens, 1), (batch_size, num_tokens, 1)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # --> (batch_size, num_tokens)
        end_logits = end_logits.squeeze(-1)  # --> (batch_size, num_tokens)
        return start_logits, end_logits


class AlbertForQA(nn.Module):
    def __init__(self, albert_path, dropout):
        super(AlbertForQA, self).__init__()
        self.qa_albert = AlbertModel.from_pretrained(albert_path)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.qa_albert.config.hidden_size, 2)

    def forward(self, input_ids, mask, token_type_ids):
        sequence_output, _ = self.qa_albert(
            input_ids=input_ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        logits = self.dropout(sequence_output)
        logits = self.fc(logits)  # (batch_size, num_tokens, 1), (batch_size, num_tokens, 1)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # --> (batch_size, num_tokens)
        end_logits = end_logits.squeeze(-1)  # --> (batch_size, num_tokens)
        return start_logits, end_logits

class BertClassifier(nn.Module):
    def __init__(self, bert_path, dropout, n_class):
        super(BertClassifier, self).__init__()
        self.bert_path = bert_path
        self.n_class = n_class
        self.bert = BertModel.from_pretrained(self.bert_path)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, self.n_class)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        out = self.dropout(out)
        out = self.fc(out)
        return out


class AlbertClassifier(nn.Module):
    def __init__(self, albert_path, dropout, n_class):
        super(AlbertClassifier, self).__init__()
        self.albert_path = albert_path
        self.n_class = n_class
        self.albert = AlbertModel.from_pretrained(self.albert_path)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.albert.config.hidden_size, n_class)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, out = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        out = self.dropout(out)
        out = self.fc(out)
        return out
