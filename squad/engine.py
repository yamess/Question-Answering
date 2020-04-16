import pickle
import torch
from tqdm import tqdm
import datetime
from squad.utils import f1_score, exact_match_score, normalize_answer
import squad.config as config

def qa_train_loop(data_loader, model, optimizer, criterion, device, scheduler=None):
    total_loss, total_correct, total_prediction, total_f1_score = 0.0, 0.0, 0.0, 0.0
    n_total = 0.0
    model.train()
    for bi, d in enumerate(tqdm(data_loader)):

        ids = d["input_ids"].to(device, dtype=torch.long)
        mask = d["mask"].to(device, dtype=torch.long)
        token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
        start_positions = d["start_positions"].to(device)
        end_positions = d["end_positions"].to(device)
        ground_truth = d["answer_text"]
        all_tokens = d["all_tokens"]

        optimizer.zero_grad()
        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions
        )
        loss = outputs[0]
        start_scores = outputs[1]
        end_scores = outputs[2]

        start_ids = torch.argmax(start_scores, dim=-1).detach().cpu()
        end_ids = torch.argmax(end_scores, dim=-1).detach().cpu()

        tokens = list(zip(*all_tokens))
        pred_text = [
            ' '.join(
                tokens[i][start_ids[i].item(): end_ids[i].item() + 1]
            ) for i in range(ids.size(0))
        ]
        loss.backward()
        optimizer.step()
        scheduler.step()

        normalise_pred_text = [normalize_answer(s) for s in pred_text]
        f1 = [f1_score(normalise_pred_text[i], ground_truth[i].lower()) for i in range(ids.size(0))]
        exact = [exact_match_score(normalise_pred_text[i], ground_truth[i].lower()) for i in range(ids.size(0))]

        total_correct += sum(exact)
        total_f1_score += sum(f1)
        total_loss += loss.item()
        n_total += ids.size(0)

    avg_loss = total_loss / len(data_loader)
    avg_exact_match = total_correct / n_total
    avg_f1_score = total_f1_score / n_total

    train_checkpoint = {
        'id': bi,
        'loss': avg_loss,
        'f1_score': avg_f1_score,
        'exact_match': avg_exact_match
    }
    with open("train_checkpoint.pkl", "wb") as f:
        pickle.dump(train_checkpoint, f)
    return avg_loss, avg_exact_match, avg_f1_score

def qa_eval_loop(data_loader, model, criterion, device):
    total_loss, total_correct, total_prediction, total_f1_score = 0.0, 0.0, 0.0, 0.0
    n_total = 0.0
    model.eval()
    with torch.no_grad():
        for bi, d in enumerate(tqdm(data_loader)):
            ids = d["input_ids"].to(device, dtype=torch.long)
            mask = d["mask"].to(device, dtype=torch.long)
            token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
            start_positions = d["start_positions"].to(device)
            end_positions = d["end_positions"].to(device)
            ground_truth = d["answer_text"]
            all_tokens = d["all_tokens"]

            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                token_type_ids=token_type_ids,
                start_positions=start_positions,
                end_positions=end_positions
            )

            loss = outputs[0]
            start_scores = outputs[1]
            end_scores = outputs[2]

            start_ids = torch.argmax(start_scores, dim=-1).detach().cpu()
            end_ids = torch.argmax(end_scores, dim=-1).detach().cpu()

            tokens = list(zip(*all_tokens))
            pred_text = [
                ' '.join(
                    tokens[i][start_ids[i].item(): end_ids[i].item() + 1]
                ) for i in range(ids.size(0))
            ]
            normalise_pred_text = [normalize_answer(s) for s in pred_text]
            # print(normalise_pred_text)
            # print(ground_truth)
            f1 = [f1_score(normalise_pred_text[i], ground_truth[i].lower()) for i in range(ids.size(0))]
            exact = [exact_match_score(normalise_pred_text[i], ground_truth[i].lower()) for i in range(ids.size(0))]
            total_correct += sum(exact)
            total_f1_score += sum(f1)
            total_loss += loss.item()
            n_total += ids.size(0)

    avg_loss = total_loss / len(data_loader)
    avg_exact_match = total_correct / n_total
    avg_f1_score = total_f1_score / n_total

    eval_checkpoint = {
        'id': bi,
        'loss': avg_loss,
        'f1_score': avg_f1_score,
        'exact_match': avg_exact_match
    }
    with open("eval_checkpoint.pkl", "wb") as f:
        pickle.dump(eval_checkpoint, f)

    return avg_loss, avg_exact_match, avg_f1_score

def format_time(elapsed):
    """
    Function to calculate the computation time of a task
    :param elapsed: 
    :return: 
    """
    elapsed_rounded = int(round(elapsed))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def qa_eval_loop_2(data_loader, model, criterion, device):
    total_loss, total_correct, total_prediction, total_f1_score = 0.0, 0.0, 0.0, 0.0
    n_total = 0.0
    f1 = list()
    match = list()

    model.eval()
    with torch.no_grad():
        for bi, d in enumerate(tqdm(data_loader)):
            ids = d["input_ids"].to(device, dtype=torch.long)
            mask = d["mask"].to(device,dtype=torch.long)
            token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
            start_positions = d["start_positions"].to(device)
            end_positions = d["end_positions"].to(device)
            ground_truth = d["answer_text"]
            all_tokens = d["all_tokens"]

            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                token_type_ids=token_type_ids,
                start_positions=start_positions,
                end_positions=end_positions
            )

            loss = outputs[0]
            start_scores = outputs[1]
            end_scores = outputs[2]

            start_ids = torch.argmax(start_scores, dim=-1).detach().cpu()
            end_ids = torch.argmax(end_scores, dim=-1).detach().cpu()

            tokens = list(zip(*all_tokens))
            pred_text = [
                ' '.join(
                    tokens[i][start_ids[i].item(): end_ids[i].item() + 1]
                ) for i in range(ids.size(0))
            ]
            normalise_pred_text = [normalize_answer(s) for s in pred_text]
            # print(normalise_pred_text)
            # print(ground_truth)
            f1 = [f1_score(normalise_pred_text[i], ground_truth[i].lower()) for i in range(ids.size(0))]
            exact = [exact_match_score(normalise_pred_text[i], ground_truth[i].lower()) for i in range(ids.size(0))]
            total_correct += sum(exact)
            total_f1_score += sum(f1)
            total_loss += loss.item()
            n_total += ids.size(0)

    avg_loss = total_loss / len(data_loader)
    avg_exact_match = total_correct / n_total
    avg_f1_score = total_f1_score / n_total

    eval_checkpoint = {
        'id': bi,
        'loss': avg_loss,
        'f1_score': avg_f1_score,
        'exact_match': avg_exact_match
    }
    with open("eval_checkpoint.pkl", "wb") as f:
        pickle.dump(eval_checkpoint, f)

    return avg_loss, avg_exact_match, avg_f1_score