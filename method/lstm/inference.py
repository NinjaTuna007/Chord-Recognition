import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import utils
import logging
from .lstm import LSTMClassifier
from preprocess import get_chord_params_by_mirex_category, gen_train_data
from .dataset import ChordDataset, split_data_to_batch

def get_model(args):
    num_classes = get_chord_params_by_mirex_category(args.category)['label_size']
    input_size = utils.get_input_size(args)
    # create model
    if args.model == 'LSTM':
        model = LSTMClassifier(input_size=input_size, hidden_dim=args.hidden_dim, output_size=num_classes, num_layers=args.num_layers, device=args.device, bidirectional=args.bidirectional, dropout=args.dropout)
        model = model.to(args.device)
    else:
        raise NotImplementedError
    return model

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

def inference(args):
    assert args.snapshot_path, 'Please specify snapshot path.'
    data = gen_train_data(args, args.data_list)
    model = get_model(args)
    utils.load_checkpoint(model, args.snapshot_path)
    model = model.to(args.device)
    data = split_data_to_batch(data, args)
    data_loader = torch.utils.data.DataLoader(ChordDataset(data), batch_size=args.batch_size, shuffle=False, num_workers=0)
    acc = validate(model, data_loader, args.device, print_results=True)
    return acc