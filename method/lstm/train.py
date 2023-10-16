import datetime
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from pprint import pformat
import utils
import logging
from .lstm import LSTMClassifier
# from preprocess.generators import gen_train_data
from preprocess import get_chord_params_by_mirex_category, feature_params, get_input_size

def split_data_to_batch(data, len_sub_audio, feature_type):
    inds_len = int(len_sub_audio * (feature_params[feature_type]['fs'] / feature_params[feature_type]['hop_length']))
    data_batch = []
    for d in data:
        audio_name, X, y = d
        # split audio to sub-audios
        X = np.array_split(X, np.ceil(len(X) / inds_len))
        y = np.array_split(y, np.ceil(len(y) / inds_len))
        assert len(X) == len(y)
        for i in range(len(X)):
            data_batch.append((audio_name, X[i], y[i]))
    # padding for X and y
    max_len = max([len(d[1]) for d in data_batch])
    for i in range(len(data_batch)):
        data_batch[i] = (data_batch[i][0], np.pad(data_batch[i][1], ((0, max_len - len(data_batch[i][1])), (0, 0)), 'constant', constant_values=((-1, -1), (-1, -1))), np.pad(data_batch[i][2], (0, max_len - len(data_batch[i][2])), 'constant', constant_values=(-1, -1)))
    return data_batch

def get_model(args):
    num_classes = get_chord_params_by_mirex_category(args.category)['label_size']
    input_size = get_input_size(args.feature_type)
    # create model
    if args.model == 'LSTM':
        model = LSTMClassifier(input_size=input_size, hidden_dim=args.hidden_dim, output_size=num_classes, num_layers=args.num_layers, device=args.device, bidirectional=args.bidirectional, dropout=args.dropout)
        model = model.to(args.device)
    else:
        raise NotImplementedError
    return model

class ChordDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        return self.data[index][1], self.data[index][2]
    
    def __len__(self):
        return len(self.data)

def get_data(args):
    # load preprocessed data
    data_list_name = args.data_list.split('/')[-1].split('.')[0]
    data_snapshot_name = '_'.join([data_list_name, args.feature_type, args.category]) + '.pt'
    data = torch.load(os.path.join(args.data_snapshot_path, data_snapshot_name))

    data = split_data_to_batch(data, args.len_sub_audio, args.feature_type)

    # split data to train and val randomly
    ind = np.arange(len(data))
    np.random.shuffle(ind)
    train_ind = ind[:int(len(data) * 0.8)]
    val_ind = ind[int(len(data) * 0.8):]
    train_dataset = ChordDataset([data[i] for i in train_ind])
    val_dataset = ChordDataset([data[i] for i in val_ind])
    
    # create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader

def train_epoch(model, train_loader, criterion, optimizer, epoch, args):
    model.train()
    losses = utils.AvgrageMeter()
    for data in train_loader:
        inputs, labels = data
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        outputs = model(inputs)
        outputs = outputs.view(-1, outputs.size(2))
        labels = labels.view(-1)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), inputs.shape[0])
    return losses.avg
        

def train(args):
    model = get_model(args)
    
    train_loader, val_loader = get_data(args)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.sch_step_size, gamma=args.sch_gamma)
    best_acc = 0

    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, criterion, optimizer, epoch, args)
        logging.info('Epoch: [{}/{}], Loss: {}'.format(epoch+1, args.epochs, loss))
        if (epoch + 1) % args.val_step == 0:
            train_acc = validate(model, train_loader, args.device)
            val_acc = validate(model, val_loader, args.device)
            logging.info('Epoch: [{}/{}], Train acc: {}, Val acc: {}'.format(epoch+1, args.epochs, train_acc, val_acc))
            if val_acc > best_acc:
                best_acc = val_acc
                # save checkpoint
                utils.save_checkpoint(model, os.path.join(args.log_path, 'best.pth'))

        # disable dropout on last 10 epochs
        if args.epochs - epoch == 10:
            model.disable_dropout()
        scheduler.step()

    logging.info('Finished Training')
    logging.info('Best val acc: {}'.format(best_acc))
    return model, best_acc


def validate(model, data_loader, device, print_results=False):
    model.eval()
    correct, total, acc = 0, 0, 0
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            predicted = outputs.topk(1, dim=2)[1].squeeze().view(-1)
            labels = labels.view(-1)
            predicted = predicted[labels >= 0]
            labels = labels[labels >= 0]
            total += len(labels)
            correct += (predicted == labels).sum().item()
    if total:
        acc = 100 * correct / total
    if print_results:
        logging.info(f'Val acc: {acc}')
    return acc
