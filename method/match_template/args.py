import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--feature_type', type=str, default='CHROMA_CQT', choices=['CHROMA_CQT', 'CHROMA_STFT'])