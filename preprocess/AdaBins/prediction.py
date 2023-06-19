from infer import InferenceHelper
import argparse
import os


def main(args):
    infer_helper = InferenceHelper(dataset=args.dataset)

    # predict depths of images stored in a directory and store the predictions in 16-bit format in a given separate dir
    infer_helper.predict_dir(args.src_dir, args.dst_dir, args.viz, args.eval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Libar')
    parser.add_argument('--src_dir', type=str, default='')
    parser.add_argument('--dst_dir', type=str, default='')
    parser.add_argument('--dataset', type=str, default='nyu')
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    assert os.path.isdir(args.src_dir)
    os.makedirs(args.dst_dir, exist_ok=True)
    main(args)