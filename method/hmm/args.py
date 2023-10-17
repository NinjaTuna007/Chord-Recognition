import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--feature_type', type=str, default='CHROMA_STFT', choices=['CHROMA_CQT', 'CHROMA_STFT'])

# Feature params
parser.add_argument('--hop_length', type=int, default=4096)