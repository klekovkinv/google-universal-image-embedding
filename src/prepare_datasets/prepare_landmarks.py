import argparse
import os
import shutil

import numpy
import pandas as pd
from tqdm import tqdm


def copy_files(l, src, dst):
    for file in l:
        src = os.path.join(src, file[0], file[1], file[2], file + '.jpg')
        dst_ = os.path.join(dst, file + '.jpg')
        shutil.copyfile(src, dst_)


def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='data/sources/landmark-recognition-2021',
                        help='path to input data folder')
    parser.add_argument('--output', type=str, default='data/dataset',
                        help='path to output folder')
    parser.add_argument('--samples', type=int, default=55,
                        help='Samples number per class')
    parser.add_argument('--num-classes', type=int, default=10000,
                        help='Number of classes to take')
    args = parser.parse_args()

    input_image_folder = os.path.join(args.input, 'train')
    input_csv_path = os.path.join(args.input, 'train.csv')
    output_train_folder = os.path.join(args.output, 'train')
    output_val_folder = os.path.join(args.output, 'val')

    # listdir ignore hidden
    start_index = len([f for f in os.listdir(output_train_folder)
                       if not f.startswith('.')]) + 1

    # read csv
    train_csv = pd.read_csv(input_csv_path)
    # find most frequent classes
    for idx, cls in tqdm(enumerate(
            train_csv['landmark_id'].value_counts().index.tolist()[:args.num_classes])):
        # create folders
        os.mkdir(os.path.join(output_train_folder, str(start_index + idx)))
        os.mkdir(os.path.join(output_val_folder, str(start_index + idx)))

        # get candidates
        candidates = train_csv[
            train_csv['landmark_id'] == cls]['id'].values.tolist()
        samples_per_class = min(len(candidates), args.samples)
        # choose samples
        chosen = numpy.random.choice(candidates,
                                     size=samples_per_class, replace=False)

        # copy files
        copy_files(chosen[:-5], input_image_folder,
                   os.path.join(output_train_folder, str(start_index + idx)))
        copy_files(chosen[-5:], input_image_folder,
                   os.path.join(output_val_folder, str(start_index + idx)))


if __name__ == '__main__':
    main()
