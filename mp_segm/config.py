import argparse
import os.path as op


def parse_args_function():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainsplit",
        type=str,
        default='train',
        choices=['train', 'minitrain', 'trainval'],
        help="You should use train by default but use minitrain to speed up coding")
    parser.add_argument(
        "--valsplit",
        type=str,
        default='val',
        choices=['val', 'minival'],
        help="You should use val by default but use minival to speed up coding")
    parser.add_argument(
        "--input_img_size",
        type=int,
        default=128,
        help="Input image size. Do not change this from its default"
    )
    parser.add_argument(
        "--load_ckpt",
        type=str,
        default="",
        help='Load pre-trained weights from your training procedure for test.py. e.g., logs/EXP_KEY/latest.pt'
    )
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=10000,
        help="Number of epochs to train"
    )
    parser.add_argument(
        "--eval_every_epoch",
        type=int,
        default=1,
        help="Evaluate your model in the training process every K training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        # default=3e-4,
        help='Learning rate'
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8
    )
    args = parser.parse_args()

    root_dir = op.join('.')
    data_dir = op.join(root_dir, 'data')
    args.input_img_shape = tuple([args.input_img_size]*2)
    args.root_dir = root_dir
    args.data_dir = data_dir
    args.experiment = None
    return args


args = parse_args_function()
