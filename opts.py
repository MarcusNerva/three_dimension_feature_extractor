import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='Experiment', help='video dataset name')
    parser.add_argument('--video_dir', default='./videos', type=str, help='Root path of input videos')
    parser.add_argument('--ext', default='mp4', type=str, help='extension name of videos')
    parser.add_argument('--model', default='./checkpoints/resnext-101-kinetics.pth', type=str, help='Model file path')
    parser.add_argument('--save_dir', default='./store', type=str, help='Output file path')
    parser.add_argument('--mode', default='feature', type=str, help='Mode (score | feature). score outputs class scores. feature outputs features (after global average pooling).')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch Size')
    parser.add_argument('--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--model_name', default='resnext', type=str, help='Currently only support resnet')
    parser.add_argument('--model_depth', default=101, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.add_argument('--step', type=int, default=5, help='argument step in dataset.make_dataset')
    parser.set_defaults(verbose=False)
    parser.add_argument('--verbose', action='store_true', help='')
    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    return args
