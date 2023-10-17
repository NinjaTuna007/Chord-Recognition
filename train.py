import torch
import utils
import os
import logging
import datetime
from pprint import pformat
from method import lstm_train
from args import get_args
logging.getLogger('numba').setLevel(logging.WARNING)

def train(args):
    assert args.method in ['lstm'], 'Only lstm method need to be trained.'
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    utils.fix_seed()
    log_path = os.path.join(args.log_path, f'{args.model}_{args.category}_{args.feature_type}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    os.makedirs(log_path, exist_ok=True)
    utils.init_logger(os.path.join(log_path, 'train.log'))
    args.log_path = log_path
    logging.info('Arguments:\n' + pformat(args.__dict__))
    model, best_acc = lstm_train(args)

if __name__ == '__main__':
    args = get_args()
    train(args)