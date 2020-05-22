# Useful functions for your implementation
from typing import List, Tuple
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler

from bpe import BytePairEncoding
from pretrain import PretrainDataset

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def pretrain_collate_fn(
    samples: List[Tuple[List[int], List[int], List[bool], bool]]
) -> torch.Tensor:
    """ Sentence collate function
    
    How to Use:
    data_loader = DataLoader(sent_dataset, ..., collate_fn=pretrain_collate_fn, ...)
    """
    PAD = BytePairEncoding.PAD_token_idx
    src, mlm, mask, nsp = zip(*samples)
    src = pad_sequence([torch.Tensor(sentence).to(torch.long) for sentence in src], padding_value=PAD)
    mlm = pad_sequence([torch.Tensor(sentence).to(torch.long) for sentence in mlm], padding_value=PAD)
    mask = pad_sequence([torch.Tensor(m).to(torch.bool) for m in mask], padding_value=False)
    nsp = torch.Tensor(nsp).to(torch.bool)

    return src, mlm, mask, nsp

def imdb_collate_fn(
    samples: List[Tuple[List[int], bool]]
) -> torch.Tensor:
    """ IMDB collate function
    
    How to Use:
    data_loader = DataLoader(imdb_dataset, ..., collate_fn=imdb_collate_fn, ...)
    """
    PAD = BytePairEncoding.PAD_token_idx
    sentences, labels = zip(*samples)
    sentences = pad_sequence([torch.Tensor(sentence).to(torch.long) for sentence in sentences], padding_value=PAD)
    labels = torch.Tensor(labels).to(torch.bool)

    return sentences, labels

class ImdbBucketSampler(Sampler):
    """ IMDB dataset bucketed batch sampler

    How to Use:
    batch_sampler = ImdbBucketSampler(imdb_dataset, batch_size, shuffle=True)
    data_loader = DataLoader(imdb_dataset, ..., batch_size=1, batch_sampler=batch_sampler, ...)
    """
    def __init__(self, dataset, batch_size, shuffle=False):
        super().__init__(dataset)
        self.shuffle = shuffle

        _, indices = zip(*sorted((len(sentences), index) for index, (sentences, _) in enumerate(dataset)))
        self.batched_indices = [indices[index: index+batch_size] for index in range(0, len(indices), batch_size)]

    def __len__(self):
        return len(self.batched_indices)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batched_indices)

        for batch in self.batched_indices:
            yield batch

def plot_values(
    train_values: List[float], 
    val_values: List[float],
    title: str
) -> None:
    x = list(range(1, len(train_values)+1))
    plt.figure()
    plt.title(title)
    plt.plot(x, train_values, marker='o', label='Training')
    plt.plot(x, val_values, marker='x', label='Validataion')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.legend()
    plt.savefig(title + '.png')
