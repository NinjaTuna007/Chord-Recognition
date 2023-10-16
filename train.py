from method.lstm.train import *

if __name__ == '__main__':
    args = get_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    utils.fix_seed()
    log_path = os.path.join(args.log_path, f'{args.model}_{args.category}_{args.feature_type}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    os.makedirs(log_path, exist_ok=True)
    utils.init_logger(os.path.join(log_path, 'train.log'))
    args.log_path = log_path
    logging.info('Arguments:\n' + pformat(args.__dict__))
    train(args)