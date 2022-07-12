import glob
import os
import re

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.utils import unique_name, merge_and_split, get_class_imbalance_weights

'''
Feature       | Labels
----------------------
Intersection     |  1
Non Intersection |  0
'''


def get_df(path, zone):
    data = list()
    path = os.path.join(path, '**', '*.jpg')
    for filepath in glob.iglob(path, recursive=True):
        temp = list()
        temp.append(filepath)  # path
        name = unique_name(filepath)

        temp.append(name)
        temp.append(zone)  # work-zone or non work zone
        # time of day i.e. Day = 1, Night =0
        if re.search('Day', filepath):
            temp.append(1)
        elif re.search('Night', filepath):
            temp.append(0)
        else:
            continue
        data.append(temp)
    df = pd.DataFrame(data, columns=['path', 'name', 'intersection', 'tod'])
    return df


def get_transform(resize_shape, jitter=0.10, split="val"):
    if split == "train":
        transform = transforms.Compose([transforms.Resize((resize_shape[0], resize_shape[1])),
                                        transforms.ColorJitter(saturation=jitter, hue=jitter,
                                                               contrast=jitter, brightness=jitter), 
                                        transforms.RandomRotation(degrees=(-60,60)),
                                        # transforms.RandomAdjustSharpness(sharpness_factor=2),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
                                        ])
        return transform
    elif split == "val" or split == "test":
        transform = transforms.Compose([transforms.Resize((resize_shape[0], resize_shape[1])),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
                                        ])
        return transform


class DatasetBaseline(Dataset):

    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image = Image.open(self.df.iloc[item, 0])
        w, h = image.size
        image = image.crop((0, 30, w, h))  # cropping watermark
        image = self.transform(image)
        label = self.df.iloc[item, 2]
        tod = self.df.iloc[item, 3]
        data = {"image": image, "label": label, "tod": tod}
        return data


def dataset_baseline(path_int, path_non_int, out_dir="output", resize_shape=(240, 360),
                     jitter=0.05, dataset=DatasetBaseline, save=True):
    """****************** Intersection ******************"""
    int_df = get_df(path_int, zone=1)

    """****************** Non Intersection ******************"""
    non_int_df = get_df(path_non_int, zone=0)

    all_data_df, train_df, test_df, val_df = merge_and_split(int_df=int_df, non_int_df=non_int_df,
                                                             out_dir=out_dir, save=save)
    weights_int, weighs_tod = get_class_imbalance_weights(train_df)

    # sanity check
    # train_test_df = test_df["name"].isin(train_df["name"])
    # train_test_df.to_csv("train_test_df.csv")
    # train_val_df = val_df["name"].isin(train_df["name"])
    # train_val_df.to_csv("train_val_df.csv")

    """Creating dataset"""
    train_set = dataset(df=train_df, transform=get_transform(resize_shape, jitter, split="train"))
    test_set = dataset(df=test_df, transform=get_transform(resize_shape, split="test"))
    val_set = dataset(df=val_df, transform=get_transform(resize_shape, split="val"))

    return train_set, test_set, val_set, weights_int, weighs_tod


if __name__ == "__main__":
    pass
