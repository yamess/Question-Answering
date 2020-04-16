import string
from collections import Counter
import torch
import re

def exact_match(start_score, end_score, start_positions, end_positions):
    start_idx = torch.max(start_score, dim=-1)[1]
    end_idx = torch.max(end_score, dim=-1)[1]
    correct = ((start_idx == start_positions).all() and (end_idx == end_positions).all()).sum().item()
    return correct

def loss_fn(o1, o2, t1, t2, criterion):
    loss_1 = criterion(o1, t1)
    loss_2 = criterion(o2, t2)
    return loss_1 + loss_2

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
