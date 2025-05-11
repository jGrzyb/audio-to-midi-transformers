import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence

import pandas as pd
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
import h5py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Tokenizer:
    def __init__(self, time_count: int = 26, note_count: int = 110, vel_count: int = 2):
        self.val_to_velo_id: dict = {i: i + 1 for i in range(vel_count)}
        self.val_to_note_id: dict = {i: i + 1 + vel_count for i in range(note_count)}
        self.val_to_time_id: dict = {i: i + 1 + vel_count + note_count for i in range(time_count)}

        self.velo_id_to_val: dict = {v: k for k, v in self.val_to_velo_id.items()}
        self.note_id_to_val: dict = {v: k for k, v in self.val_to_note_id.items()}
        self.time_id_to_val: dict = {v: k for k, v in self.val_to_time_id.items()}
        
        self.id_to_token: dict = {
            **{self.val_to_velo_id[i]: f'velo_{i}' for i in self.val_to_velo_id},
            **{self.val_to_note_id[i]: f'note_{i}' for i in self.val_to_note_id},
            **{self.val_to_time_id[i]: f'time_{i}' for i in self.val_to_time_id},
            0: '<pad>',
            vel_count + note_count + time_count + 1: '<bos>',
            vel_count + note_count + time_count + 2: '<eos>'
        }
        
        self.token_to_id: dict = {v: k for k, v in self.id_to_token.items()}
    

    def tuple_to_ids(self, tuple: tuple):
        return [self.val_to_time_id[tuple[0]], self.val_to_note_id[tuple[1]], self.val_to_velo_id[tuple[2]]]
    

    def tuple_list_to_ids(self, tuple_list: list[tuple]):
        l = []
        for t in tuple_list:
            l.extend(self.tuple_to_ids(t))
        return l


    def id_list_to_tuple_list(self, id_list: list[int]):
        l = []
        for i in range(0, len(id_list), 3):
            if i + 3 > len(id_list):
                break
            t = []
            for j, d in enumerate([self.time_id_to_val, self.note_id_to_val, self.velo_id_to_val]):
                if min(d) <= id_list[i+j] <= max(d):
                    t.append(d[id_list[i+j]])
                else:
                    t.append(-1)
            l.append(tuple(t))
        return l


def get_max_length(df: pd.DataFrame, verbose: bool = False):
    max_len = 0
    for i, (idx, row) in enumerate(df.iterrows()):
        with open(row['midi']) as f:
            lines = f.readlines()
            max_len = max(max_len, len(lines))
        if verbose and i * 100 % len(df) == 0:
            print(f"Processed {i} / {len(df)} files")
    return max_len * 3


class ProgressBar:
    def __init__(self, total, length=40):
        self.total = total
        self.length = length
        self.current = 0

    def update(self, step=1):
        self.current += step
        progress = self.current / self.total
        filled_length = int(self.length * progress)
        bar = '=' * filled_length + '-' * (self.length - filled_length)
        progress = self.current / self.total
        filled_length = int(self.length * progress)
        bar = '=' * filled_length + '-' * (self.length - filled_length)
        if self.current == 1 or self.current == self.total or filled_length > int(self.length * ((self.current - step) / self.total)):
            print(f'\r|{bar}| {self.current}/{self.total} ({progress:.2%})', end='')

    def finish(self):
        self.update(self.total - self.current)
        print()

csv_path = 'dataset1.csv'
folder_path = 'dataset1'
model_name = 'model1'
h5_path = f'dataset1.h5'

df = pd.read_csv(csv_path)
df['image'] = df['image'].apply(lambda x: os.path.join(folder_path, x))
df['midi'] = df['midi'].apply(lambda x: os.path.join(folder_path, x))
# df = df.sample(100)

max_len = get_max_length(df) + 10
max_len

transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

tokenizer = Tokenizer()


t = time.time()
progress_bar = ProgressBar(len(df))
for i, (idx, row) in enumerate(df.iterrows()):
    image_path = row['image']
    midi_path = row['midi']
    song_name = image_path.split('/')[-1].split('.')[0]

    image = Image.open(image_path).convert('L')
    # image = transform(image)

    midi = pd.read_csv(midi_path)
    midi['time'] = midi['time'] // 100
    midi['velocity'] = (midi['velocity'] > 0).astype(int)
    midi = midi.values.tolist()
    midi = tokenizer.tuple_list_to_ids(midi)
    midi.insert(0, tokenizer.token_to_id['<bos>'])
    midi.append(tokenizer.token_to_id['<eos>'])
    midi.extend([tokenizer.token_to_id['<pad>']] * (max_len - len(midi)))
    progress_bar.update()
print(time.time() - t)


# t = time.time()
# progress_bar = ProgressBar(len(df))
# for i, (idx, row) in enumerate(df.iterrows()):
#     image_path = row['image']
#     midi_path = row['midi']
#     song_name = image_path.split('/')[-1].split('.')[0]

#     image = Image.open(image_path).convert('L')
#     image = transform(image)

#     midi = pd.read_csv(midi_path)
#     midi['time'] = midi['time'] // 100
#     midi['velocity'] = (midi['velocity'] > 0).astype(int)
#     midi = midi.values.tolist()
#     # midi = tokenizer.tuple_list_to_ids(midi)
#     # midi.insert(0, tokenizer.token_to_id['<bos>'])
#     # midi.append(tokenizer.token_to_id['<eos>'])
#     # midi.extend([tokenizer.token_to_id['<pad>']] * (max_len - len(midi)))
#     with h5py.File(h5_path, 'a') as h5_file:
#         if song_name not in h5_file:
#             song_group = h5_file.create_group(song_name)
#         else:
#             song_group = h5_file[song_name]

#         chunk_idx = len(song_group)
#         chunk_group = song_group.create_group(f'chunk_{chunk_idx}')
#         chunk_group.create_dataset('image', data=image.numpy(), compression='gzip')
#         chunk_group.create_dataset('midi', data=np.array(midi), compression='gzip')
#     progress_bar.update()
# print(time.time() - t)


# t = time.time()
# with h5py.File(h5_path, 'r+') as h5_file:
#     progress_bar = ProgressBar(len(h5_file.keys()))
#     for i, song_name in enumerate(h5_file.keys()):
#         song_group = h5_file[song_name]
#         for chunk_name in song_group.keys():
#             chunk_group = song_group[chunk_name]
#             midi = chunk_group['midi'][:]

#             # Perform previously commented actions
#             midi = tokenizer.tuple_list_to_ids(midi)
#             midi.insert(0, tokenizer.token_to_id['<bos>'])
#             midi.append(tokenizer.token_to_id['<eos>'])
#             midi.extend([tokenizer.token_to_id['<pad>']] * (max_len - len(midi)))
#         progress_bar.update()

# print(time.time() - t)