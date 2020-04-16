import copy
import os
import pandas as pd
import pickle
import time

import torch
import torch.nn as nn
import transformers
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer,BertForQuestionAnswering, \
    get_linear_schedule_with_warmup

import squad.config as config
from squad.model import BertForQA
from squad.dataset import QADataset
from squad.engine import qa_eval_loop, qa_train_loop,format_time


def run(model):
    df_train = pd.read_csv("data/data_train.csv", engine="python", encoding="utf-8")
    # df_train = df_train_.loc[:2000, :]

    df_valid = pd.read_csv("data/data_valid.csv", engine="python", encoding="utf-8")
    # df_valid = df_valid_.loc[:500, :]
    tokenizer = BertTokenizer.from_pretrained(config.BERT_TOK_PATH, lower=True)

    # Train Preprocessing
    train_datset = QADataset(
        question=df_train.question.values,
        context=df_train.context.values,
        start_positions=df_train.answer_start,
        end_positions=df_train.answer_end,
        answer_text=df_train.answer_text,
        tokenizer=tokenizer,
        max_len=config.MAX_LEN
    )
    train_data_loader = DataLoader(
        dataset=train_datset,
        sampler=RandomSampler(train_datset),
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=False
    )

    # Validation data preprocessing
    validation_dataset = QADataset(
        question=df_valid.question.values,
        context=df_valid.context.values,
        start_positions=df_valid.answer_start,
        end_positions=df_valid.answer_end,
        answer_text=df_valid.answer_text,
        tokenizer=tokenizer,
        max_len=config.MAX_LEN
    )
    validation_data_loader = DataLoader(
        dataset=validation_dataset,
        sampler=SequentialSampler(validation_dataset),
        batch_size=config.VALID_BATCH_SIZE
    )

    model.to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    criterion.to(config.DEVICE)

    optimizer = transformers.AdamW(model.parameters(), lr=config.LR)
    total_steps = len(train_data_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    best_valid_f1_score = 0  # We st the valid to an higher value in order
    best_epoch = 0

    # Model Output and path setup

    if os.path.exists(config.BERT_MODEL_OUTPUT):
        print("Loading model from last checkpoint...")
        state = torch.load(config.BERT_MODEL_OUTPUT)
        model.load_state_dict(state['best_state_dict'])
        best_valid_f1_score = state['best_valid_f1_score']
        best_epoch = state['epoch']
        print(f"Best Validations Accuracy so far: {best_valid_f1_score:.3f} at Epoch {best_epoch}\n")

    # Tensorboard writer to write logs
    writer = SummaryWriter(config.LOG_DIR)

    # Training loop over the numver of epochs defined
    training_stats = []
    for epoch in range(config.EPOCHS):
        real_epoch = best_epoch + epoch
        t0 = time.time()

        train_loss, train_exact_match, train_f1_score = qa_train_loop(
            train_data_loader,
            model,
            optimizer,
            criterion,
            config.DEVICE,
            scheduler
        )

        valid_loss, valid_exact_match, valid_f1_score = qa_eval_loop(
            validation_data_loader,
            model,
            criterion,
            config.DEVICE
        )

        training_time = format_time(time.time() - t0)

        if valid_f1_score > best_valid_f1_score:
            best_valid_f1_score = valid_f1_score
            best_valid_exact_match = valid_exact_match
            best_state_dict = copy.deepcopy(model.state_dict())
            msg = (
                f"Epoch {real_epoch: <{2}}| Train loss {train_loss:5.3f} | "
                f"Train Exact Match {train_exact_match:5.3f} | Train F1 Score {train_f1_score:5.3f} | "
                f"Valid loss {valid_loss:5.3f} | Valid Exact Match {valid_exact_match:5.3f} | "
                f"Valid F1 Score {valid_f1_score:5.3f} | Elapsed: {training_time:<{8}}| +"
            )
            print(msg)

            # Let's create the checkpoint data to save
            checkpoint = {
                'epoch': real_epoch,
                'best_valid_f1_score': best_valid_f1_score,
                'best_valid_exact_match': best_valid_exact_match,
                'best_valid_loss': valid_loss,
                'best_state_dict': best_state_dict
            }
            torch.save(checkpoint, "model.pt")  # config.BERT_MODEL_OUTPUT
        else:
            msg = (
                f"Epoch {real_epoch: <{2}}| Train loss {train_loss:5.3f} | "
                f"Train Exact Match {train_exact_match:5.3f} | Train F1 Score {train_f1_score:5.3f} | "
                f"Valid loss {valid_loss:5.3f} | Valid Exact Match {valid_exact_match:5.3f} | "
                f"Valid F1 Score {valid_f1_score:5.3f} | Elapsed: {training_time: <{8}}|"
            )
            print(msg)

        # Save the training statistique
        training_stats.append(
            {
                'epoch': epoch,
                'Training_Loss': train_loss,
                'Training_F1_Score': train_f1_score,
                'Training_Exact_Match': train_exact_match,
                'Valid. Loss': valid_loss,
                'Valid_F1_Score': valid_f1_score,
                'Valid_Exact_Match': valid_exact_match,
                'Validation Time': training_time
            })

        # Save the tensorboard logs for graph
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", valid_loss, epoch)

        writer.add_scalar("F1 Score/Train", train_f1_score, epoch)
        writer.add_scalar("F1 Score/Validation", valid_f1_score, epoch)

        writer.add_scalar("Accuracy/Train", train_exact_match, epoch)
        writer.add_scalar("Accuracy/Validation", valid_exact_match, epoch)

    writer.flush()

    # Save the training stats in file for deep analysis
    with open("training_stats.pkl", "wb") as f:
        pickle.dump(training_stats, f)

    print(f"The best Model F1 Score {best_valid_f1_score}")
    print("The best Model has been saved")


if __name__ == "__main__":
    torch.cuda.empty_cache()  # Empty Cuda caches

    model = BertForQuestionAnswering.from_pretrained(config.BERT_MODEL_PATH)
    # model = BertForQA(config.BERT_MODEL_PATH, config.DROPOUT)

    print(
        f"Nbr of parameters before freezing bert layers: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    for name, param in model.named_parameters():
        if name.startswith('bert'):
            param.requires_grad = False
        # print(name)

    print(
        f"Nbr of parameters before freezing bert layers: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    run(model)
