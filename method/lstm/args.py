import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_list', type=str, default='./data/data_list/TheBeatles_train.list')
parser.add_argument('--len_sub_audio', default=40, type=int)
parser.add_argument('--log_path', default='log', type=str)
parser.add_argument('--snapshot_path', type=str)
parser.add_argument('--feature_type', type=str, default='CQT', choices=['CQT', 'STFT', 'MFCC', 'MEL', 'CHROMA_CQT', 'CHROMA_STFT'])
parser.add_argument('--model', type=str, required=True)

parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--hidden_dim', default=50, type=int)
parser.add_argument('--num_layers', default=2, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--bidirectional', default=True, type=bool)
parser.add_argument('--sch_step_size', default=100, type=int)
parser.add_argument('--sch_gamma', default=0.1, type=float)
parser.add_argument('--val_step', default=1, type=int)
parser.add_argument('--momentum', default=0.8, type=float, help='SGD momentum')
parser.add_argument('--dropout', default=(0.4, 0.0, 0.0), type=float, nargs=3, help='list of dropout values: before rnn, inside rnn, after rnn')

# Feature params
parser.add_argument('--hop_length', type=int, default=512, help='hop length')
parser.add_argument('--n_fft', type=int, default=2048, help='number of fft bins')
parser.add_argument('--n_mfcc', type=int, default=20, help='number of MFCCs')
parser.add_argument('--n_bins', type=int, default=84, help='number of bins for CQT')
parser.add_argument('--n_mels', type=int, default=128, help='number of bins for MEL')