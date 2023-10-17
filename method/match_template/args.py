import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--feature_type', type=str, default='CHROMA_CQT', choices=['CHROMA_CQT', 'CHROMA_STFT'])
parser.add_argument('--threshold', type=float, default=0.1)

# Feature params
parser.add_argument('--hop_length', type=int, default=4096)
