import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--feature_type', type=str, default='CHROMA_STFT', choices=['CHROMA_CQT', 'CHROMA_STFT'])

# Feature params
parser.add_argument('--hop_length', type=int, default=4096)

# Baum-Welch params
parser.add_argument('--max_iters', type=int, default=100, help='max iterations for Baum-Welch')
parser.add_argument('--tol', type=float, default=1e-3, help='tolerance for Baum-Welch')