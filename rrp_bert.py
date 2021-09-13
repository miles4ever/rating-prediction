import torch
import argparse
from transformers import BertTokenizer
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW
import wandb
import glob


class AmazonDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def get_dataset(args):
    if os.path.exists(args.processed_data_dir):
        train_processed = dict(np.load(os.path.join(args.processed_data_dir, "train_bert.npz")))
        valid_processed = dict(np.load(os.path.join(args.processed_data_dir, "valid_bert.npz")))
        test_processed = dict(np.load(os.path.join(args.processed_data_dir, "test_bert.npz")))

        train_labels = train_processed["labels"].astype(np.float32)
        valid_labels = valid_processed["labels"].astype(np.float32)
        del train_processed["labels"]
        del valid_processed["labels"]

        train_encodings = train_processed
        valid_encodings = valid_processed
        test_encodings = test_processed

        print(f"=> Loaded pre-processed data from {args.processed_data_dir}!")

    else:
        # Load and preprocess data
        total_train_df = pd.read_json("./train.json", lines=True)
        test_df = pd.read_json("./test.json", lines=True)
        train_df, valid_df = train_test_split(total_train_df, test_size=0.2, random_state=123)

        all_dfs = [train_df, valid_df, test_df]
        fillna_fields = ["reviewText", "summary"]
        del_fields = ["reviewTime", "unixReviewTime", "reviewHash", "image"]

        for df in all_dfs:
            # Fill N/A text field with an empty string
            for field in fillna_fields:
                df[field].fillna("", inplace=True)

            # Delete some unecessary fields
            for field in del_fields:
                del df[field]

            # use `summary` and `reviewText` field as features
            df['reviewAll'] = df['summary'] + ' ' + df['reviewText']
            del df['reviewText']
            del df['summary']

        train_labels = train_df.overall.to_numpy().astype(np.float32)
        valid_labels = valid_df.overall.to_numpy().astype(np.float32)

        # Tokenize the text, truncate to maximum length 512
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        train_encodings = tokenizer(train_df.reviewAll.tolist(), truncation=True, padding=True)
        valid_encodings = tokenizer(valid_df.reviewAll.tolist(), truncation=True, padding=True)
        test_encodings = tokenizer(test_df.reviewAll.tolist(), truncation=True, padding=True)

        print("=> Loaded and pre-processed data!")

        train_processed = dict(train_encodings)
        train_processed.update(labels=train_labels)
        valid_processed = dict(valid_encodings)
        valid_processed.update(labels=valid_labels)
        test_processed = dict(test_encodings)

        os.makedirs(args.processed_data_dir)
        np.savez(os.path.join(args.processed_data_dir, "train_bert.npz"), **train_processed)
        np.savez(os.path.join(args.processed_data_dir, "valid_bert.npz"), **valid_processed)
        np.savez(os.path.join(args.processed_data_dir, "test_bert.npz"), **test_processed)

        print(f"=> Save pre-processed data to {args.processed_data_dir}!")

    train_dataset = AmazonDataset(train_encodings, train_labels)
    valid_dataset = AmazonDataset(valid_encodings, valid_labels)
    test_dataset = AmazonDataset(test_encodings)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)  # already shuffled
    valid_loader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def get_model(args):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
    print("=> Loaded pretrained model!")
    return model


def get_optimizer(args, model):
    optimizer = AdamW(model.parameters(), lr=args.lr)
    return optimizer

def init_logger(args):
    wandb.init(project="rate-prediction",
               name=args.exp_name, config=args, resume=True, id=args.exp_name,
               dir=args.ckpt_dir)

def finetune(args):
    args.ckpt_dir = os.path.join(args.ckpt_dir, args.exp_name)
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    init_logger(args)
    train_loader, valid_loader, test_loader = get_dataset(args)
    model = get_model(args)
    model.to(args.device)
    optim = get_optimizer(args, model)


    ckpt_path_all = glob.glob(os.path.join(args.ckpt_dir, "model-*.pt"))
    if len(ckpt_path_all) > 0:
        ckpt_steps = [int(name.split("model-")[1].split(".pt")[0]) for name in ckpt_path_all if "model-best.pt" not in name]
        ckpt_path_last = ckpt_path_all[np.argmax(ckpt_steps)]

        checkpoint = torch.load(ckpt_path_last, map_location='cpu')
        step = checkpoint['step']
        min_eval_loss = checkpoint['min_eval_loss']
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        print("=> Loaded checkpoint from '{}' (step {})".format(ckpt_path_last, step))
        if step >= (args.max_epochs * len(train_loader)):
            print(f"=> Finished training (step {step})!")
            return
        start_epoch = step // len(train_loader)
    else:
        step = 0
        min_eval_loss = np.inf
        start_epoch = 0

    model.train()
    for epoch in range(start_epoch+1, args.max_epochs+1):
        pbar = tqdm(train_loader)
        skip_steps = step % len(train_loader)  # only used for starting from checkpoints

        for idx, batch in enumerate(pbar):
            if idx < skip_steps:
                continue

            optim.zero_grad()
            input_ids = batch['input_ids'].to(args.device).long()
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            pbar.set_description("Epoch {}, Loss {:.3f}".format(epoch, loss.item()))
            loss.backward()
            optim.step()

            step += 1
            if step % args.log_steps == 0:
                wandb.log({"train/loss": loss.item(), 'optim/lr': optim.param_groups[0]['lr']}, step=step)

            if step % args.eval_steps == 0:
                eval_loss = eval(step, args, model, valid_loader)
                wandb.log({"eval/loss": eval_loss}, step=step)

                to_save = {
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'eval_loss': eval_loss,
                    'min_eval_loss': min_eval_loss,
                }

                save_path = os.path.join(args.ckpt_dir, f"model-{step}.pt")
                torch.save(to_save, save_path)
                print("=> Save checkpoint to '{}' (step {})".format(save_path, step))

                if eval_loss < min_eval_loss:
                    min_eval_loss = eval_loss
                    save_path = os.path.join(args.ckpt_dir, "model-best.pt")
                    torch.save(to_save, save_path)
                    print("=> Save checkpoint to '{}' (step {})".format(save_path, step))

                print("\n")

                model.train()

    eval_loss = eval(step, args, model, valid_loader)
    to_save = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'eval_loss': eval_loss,
        'min_eval_loss': min_eval_loss,
    }

    save_path = os.path.join(args.ckpt_dir, f"model-{step}.pt")
    torch.save(to_save, save_path)
    print("=> Save checkpoint to '{}' (step {})".format(save_path, step))
    print(f"=> Finished training (step {step})!")


def eval(step, args, model, valid_loader):
    model.eval()
    losses = []

    pbar = tqdm(valid_loader)
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(args.device).long()
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            pbar.set_description("Eval Loss {:.3f}".format(loss.item()))

            losses.append(loss.item())

    eval_loss = np.mean(losses)
    print('=> Step: {} Eval loss: {:.6f}'.format(step, eval_loss))

    return eval_loss


def predict(args):
    train_loader, valid_loader, test_loader = get_dataset(args)
    model = get_model(args)
    model.to(args.device)

    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    step = checkpoint['step']
    min_eval_loss = checkpoint['min_eval_loss']
    model.load_state_dict(checkpoint['model_state_dict'])
    print("=> Loaded checkpoint from '{}' (step {})".format(args.ckpt_path, step))

    model.eval()
    # eval(step, args, model, valid_loader)

    pbar = tqdm(test_loader)
    prediction_array = []
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(args.device).long()
            attention_mask = batch['attention_mask'].to(args.device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs[0].squeeze(1).cpu().numpy()
            prediction_array += list(logits)

    # Write result
    predictions = open('rating_predictions_bert.csv', 'w')
    for idx, l in enumerate(open('rating_pairs.csv')):
        if l.startswith('userID'):
            # header
            predictions.write(l)
            continue
        u, p = l.strip().split('-')
        predict = prediction_array[idx - 1]
        predictions.write(u + '-' + p + ',' + str(predict) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data_dir", type=str, default="./processed")

    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--eval_batch_size", type=int, default=12)


    subparsers = parser.add_subparsers(title="commands", dest="command")

    finetune_cmd = subparsers.add_parser(
        "finetune",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Finetune a pretrained BERT model.")
    finetune_cmd.add_argument("--lr", type=float, default=5e-5)
    finetune_cmd.add_argument("--max_epochs", type=int, default=3)
    finetune_cmd.add_argument("--eval_steps", type=int, default=5000)
    finetune_cmd.add_argument("--log_steps", type=int, default=10)
    finetune_cmd.add_argument("--ckpt_dir", type=str,
                        default="/checkpoint/ywu/")
    finetune_cmd.add_argument("--exp_name", type=str, default="bert_finetune")

    predict_cmd = subparsers.add_parser(
        "predict",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Predict using a finetuned BERT model.")
    predict_cmd.add_argument("--ckpt_path", type=str, default="")



    args = parser.parse_args()
    if torch.cuda.is_available():
        args.device = torch.device('cuda') 
        print('=> computing on GPU: {}'.format(torch.cuda.get_device_name(0)))
    else:
        args.device = torch.device('cpu')
        print('=> computing on CPU')

    if args.command == "finetune":
        finetune(args)
    elif args.command == "predict":
        predict(args)
    else:
        raise NotImplementedError