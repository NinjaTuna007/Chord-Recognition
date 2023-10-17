import argparse
import method

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='lstm', type=str, choices=['lstm', 'match_template', 'hmm'])
    parser.add_argument('--audio_path', default='data/audio', type=str)
    parser.add_argument('--gt_path', default='data/gt', type=str)
    parser.add_argument('--category', default='MirexMajMin', type=str, choices=['MirexMajMin'])
    parser.add_argument('--data_list', type=str, default='./data/data_list/TheBeatles.list', help='data list for inference')
    parser.add_argument('--data_snapshot_path', default='data/snapshot', type=str)
    parser.add_argument('--sr', type=int, default=44100)

    args, remaining_argv = parser.parse_known_args()
    
    method_parser = method.__dict__[f'{args.method.lower()}_parser']
    method_args, _ = method_parser.parse_known_args(remaining_argv)

    merged_args = argparse.Namespace(**vars(args), **vars(method_args))

    return merged_args