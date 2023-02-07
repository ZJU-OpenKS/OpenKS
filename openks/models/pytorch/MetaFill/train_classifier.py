import torch
import numpy as np
from transformers import BertTokenizer, GPT2Tokenizer
from torch import nn
from transformers import BertModel, GPT2Model
import json
import os 
import argparse
from classifier import GPTClassifier
from torch.optim import Adam
from tqdm import tqdm


class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_name, edge_text=None, tokenizer=None, labels=None):
        with open(file_name, "r") as f:
            lines = f.read().splitlines()
        with open(edge_text, "r") as f:
            edge_text = json.load(f)
        self.tokenizer = tokenizer
        self.h_labels = []
        self.t_labels = []
        self.h_texts = []
        self.t_texts = []
        self.contexts = []
        for l in lines:
            head, r, tail = l.split("\t")
            th, h = head.split(" : ")
            tt, t = tail.split(" : ")
            h_text = tokenizer(h, padding="max_length", max_length = 200, truncation=True, return_tensors="pt")
            t_text = tokenizer(t, padding="max_length", max_length = 200, truncation=True, return_tensors="pt")
            context = tokenizer(h + " [SEP] " + edge_text[r] + " [SEP] " + t, padding="max_length", max_length = 200, truncation=True, return_tensors="pt")
            if h_text['input_ids'].shape[1] == 0 or t_text['input_ids'].shape[1] == 0 or context['input_ids'].shape[1] == 0:
                continue
            self.h_labels.append(labels[th])
            self.t_labels.append(labels[tt])
            self.h_texts.append(h_text)
            self.t_texts.append(t_text)
            self.contexts.append(context)

    def classes(self):
        return self.h_labels

    def __len__(self):
        return len(self.h_labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.h_labels[idx]), np.array(self.t_labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.h_texts[idx], self.t_texts[idx]
    
    def get_context(self, idx):
        return self.contexts[idx]

    def __getitem__(self, idx):

        batch_texts_h, batch_texts_t = self.get_batch_texts(idx)
        batch_y_h, batch_y_t = self.get_batch_labels(idx)
        batch_context = self.get_context(idx)

        return batch_texts_h, batch_texts_t, batch_y_h, batch_y_t, batch_context


def train(model, train_data, val_data, learning_rate, epochs, edge_text=None, label_id=None, model_name=None):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    with open(label_id, "r") as f:
        labels = json.load(f)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    train, val = Dataset(train_data, edge_text, tokenizer, labels), Dataset(val_data, edge_text, tokenizer, labels)
    
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)
    print("load data")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    best_acc = 0
    patience = 3
    cnt = 0
    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        total_len = 0
        pbar = tqdm(train_dataloader)
        model.train()
        
        for train_input_h, train_input_t, train_label_h, train_label_t, train_context in pbar:
            train_label_h = train_label_h.to(device)
            train_label_t = train_label_t.to(device)
            mask_h = train_input_h['attention_mask'].to(device)
            input_id_h = train_input_h['input_ids'].squeeze(1).to(device)
            mask_t = train_input_t['attention_mask'].to(device)
            input_id_t = train_input_t['input_ids'].squeeze(1).to(device)
            mask_context = train_context['attention_mask'].to(device)
            context = train_context['input_ids'].squeeze(1).to(device)
            output_h, output_t = model(input_id_h, input_id_t, mask_h, mask_t, context, mask_context)                
            batch_loss_h = criterion(output_h, train_label_h)
            batch_loss_t = criterion(output_t, train_label_t)
            batch_loss = batch_loss_h + batch_loss_t
            total_loss_train += batch_loss.item()                
            acc_h = (output_h.argmax(dim=1) == train_label_h).sum().item()
            acc_t = (output_t.argmax(dim=1) == train_label_t).sum().item()
            total_len += train_label_t.shape[0]
            total_acc_train += acc_t
            pbar.set_description("acc: %.3f" %(total_acc_train / total_len))
            model.zero_grad()
            batch_loss.backward()
            optimizer.step() 
        
        total_acc_val = 0
        total_loss_val = 0
        total_len_val = 0
        pbar = tqdm(val_dataloader)
        model.eval()
        with torch.no_grad():
            for val_input_h, val_input_t, val_label_h, val_label_t, val_context in pbar:
                val_label_h = val_label_h.to(device)
                val_label_t = val_label_t.to(device)
                mask_h = val_input_h['attention_mask'].to(device)
                input_id_h = val_input_h['input_ids'].squeeze(1).to(device)
                mask_t = val_input_t['attention_mask'].to(device)
                input_id_t = val_input_t['input_ids'].squeeze(1).to(device)
                mask_context = val_context['attention_mask'].to(device)
                context = val_context['input_ids'].squeeze(1).to(device)
                output_h, output_t = model(input_id_h, input_id_t, mask_h, mask_t, context, mask_context)                
                batch_loss_h = criterion(output_h, val_label_h)
                batch_loss_t = criterion(output_t, val_label_t)
                batch_loss = batch_loss_h + batch_loss_t
                total_loss_val += batch_loss.item()                
                acc_h = (output_h.argmax(dim=1) == val_label_h).sum().item()
                acc_t = (output_t.argmax(dim=1) == val_label_t).sum().item()
                total_len_val += val_label_t.shape[0]
                total_acc_val += acc_t
                pbar.set_description("acc: %.3f" %(total_acc_val / total_len_val))
            
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / (total_len * 2): .3f} \
            | Train Accuracy: {total_acc_train / total_len: .3f} \
            | Val Loss: {total_loss_val / (total_len_val * 2): .3f} \
            | Val Accuracy: {total_acc_val / total_len_val: .3f}')
        if (total_acc_val / total_len_val) > best_acc:
            torch.save(model, model_name)
            best_acc = total_acc_val / total_len_val
            cnt = 0
        else:
            cnt += 1
        if cnt >= patience:
            break

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--train-data', type=str, default="../data_metapath/nell/train_edges.txt")
    parser.add_argument('--valid-data', type=str, default="../data_metapath/nell/valid_edges.txt")
    parser.add_argument('--edge-text', type=str, default="../data_metapath/nell/edge_text.json")
    parser.add_argument('--label-id', type=str, default="../data_metapath/nell/label_id.json")
    parser.add_argument('--model-name', type=str, default="classifier_gpt_nell")
    parser.add_argument('--class-num', type=int, default=281)
    args = parser.parse_args()
    model = GPTClassifier(n_class=args.class_num)
    train(model, args.train_data, args.valid_data, args.lr, args.epochs, args.edge_text, args.label_id, args.model_name)
