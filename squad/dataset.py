import torch


class QADataset:
    def __init__(self,
                 question,
                 context,
                 start_positions=None,
                 end_positions=None,
                 answer_text=None,
                 max_len=512,
                 tokenizer=None):
        """
        Class for Text data preparation for Question answering task
        :param question: Question text
        :param context: Context text
        :param start_positions: Ground truth start position
        :param end_positions: Ground truth end position
        :param max_len: Max length for padding
        :param tokenizer: Text tokenizer
        """
        self.answer_text = answer_text
        self.context = context
        self.question = question
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.start_positions = start_positions
        self.end_positions = end_positions

    def __len__(self):
        return len(self.context)

    def __getitem__(self, item):
        context = str(self.context[item])
        question = str(self.question[item])
        answer_text = str(self.answer_text[item])

        input_dict = self.tokenizer.encode_plus(
            question,
            context,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True
        )

        input_ids = input_dict["input_ids"]
        token_type_ids = input_dict["token_type_ids"]   # This alos represent what we call segment
        mask = input_dict["attention_mask"]
        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        out = {
            "input_ids": torch.tensor(input_ids),
            "mask": torch.tensor(mask),
            "token_type_ids": torch.tensor(token_type_ids),
            "start_positions": torch.tensor(self.start_positions[item]),
            "end_positions": torch.tensor(self.end_positions[item]),
            "answer_text": answer_text,
            "all_tokens": all_tokens
        }
        return out

class TextClfDataset:
    def __init__(self, text, tokenizer, label, max_len=512):
        """
        Class for preparing text data for text classification / sentiment analysis task for transformers models
        :param text: Text to process
        :param tokenizer: Text tokenizer
        :param max_len: The max len for the padding
        :param label: The label to predict in the task
        """
        self.text = text
        self.label = label
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len
        )

        input_ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        # padding
        padding_len = self.max_len - len(input_ids)
        input_ids = input_ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)

        out = {
            "input_ids": torch.tensor(input_ids),
            "mask": torch.tensor(mask),
            "token_type_ids": torch.tensor(token_type_ids),
            "label": torch.tensor(self.label[item])
        }
        return out
