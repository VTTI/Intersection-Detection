import os
import random
import re
from ast import literal_eval

import pandas as pd
import torch
import yaml


def get_class_imbalance_weights(train_df):
    """Class imbalance mitigation"""
    count_intersection = train_df.groupby('intersection').size().to_list()  # managing class imbalance
    count_tod = train_df.groupby('tod').size().to_list()  # managing class imbalance
    weights_intersection = torch.FloatTensor(
        [max(count_intersection) / count_intersection[0], max(count_intersection) / count_intersection[1]])
    print(weights_intersection.shape)
    weighs_tod = torch.FloatTensor([max(count_tod) / count_tod[0], max(count_tod) / count_tod[1]])
    return weights_intersection, weighs_tod


def unique_name(path):
    name = re.findall("_([0-9a-z]+)_", path.split(os.sep)[-1])[0]  # unique vehicle number
    return name


def split(len_dataset, p_train=0.80, p_test=0.10, p_val=0.10):
    len_train = int(len_dataset * p_train)
    len_test = int(len_dataset * p_test)
    len_val = int(len_dataset * p_val)

    if len_dataset == len_train + len_test + len_val:
        return len_train, len_test, len_val
    else:
        difference = len_dataset - (len_train + len_test + len_val)
        return len_train, len_test + difference, len_val


def merge_and_split(int_df, non_int_df, out_dir="output", save=True):
    """Merging both datasets"""
    all_data_df = pd.concat([int_df, non_int_df], ignore_index=True, sort=False)
    unique_values = list(all_data_df['name'].unique())
    random.Random(42).shuffle(unique_values)
    train_split, test_split, val_split = split(len(unique_values))

    train_df = all_data_df.loc[all_data_df['name'].isin(unique_values[0:train_split])]
    test_df = all_data_df.loc[all_data_df['name'].isin(unique_values[train_split:train_split + test_split])]
    val_df = all_data_df.loc[all_data_df['name'].isin(unique_values[train_split + test_split:])]

    if save:
        path = os.path.join(out_dir, 'csv')
        os.makedirs(path, exist_ok=True)
        all_data_df.to_csv(os.path.join(path, 'all_data.csv'))
        train_df.to_csv(os.path.join(path, 'train.csv'))
        test_df.to_csv(os.path.join(path, 'test.csv'))
        val_df.to_csv(os.path.join(path, 'val.csv'))
        print("Saving csv files...")

    return all_data_df, train_df, test_df, val_df


def get_configs(path):
    with open(path) as out:
        configs = yaml.load(out, Loader=yaml.FullLoader)

    for key, value in configs.items():
        print(key, ": ", value)

    int_dir = configs["INT_DIR"]
    non_int_dir = configs["NONINT_DIR"]
    output_dir = configs["OUTPUT_DIR"]
    model_name = configs["MODEL"]
    backbone = configs["BACKBONE"]
    epochs = configs["EPOCHS"]
    lr = configs["LR"]
    custom = configs["CUSTOM_IMAGES_PATH"]
    resize_shape = literal_eval(configs["RESIZE_SHAPE"])
    optimizer = configs["OPTIMIZER"]
    batch_size = configs["BATCH_SIZE"]
    log_step = configs["LOG_STEP"]

    # create output directory
    os.makedirs(output_dir, exist_ok=True)

    return [int_dir, non_int_dir, output_dir,
            model_name, backbone, epochs, lr,
            resize_shape, optimizer, batch_size, log_step,
            custom]
