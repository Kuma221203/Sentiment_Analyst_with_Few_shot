from typing import Dict, Iterator, List, Tuple
from torch import Tensor
import random
import torch
from torch.utils.data import DataLoader
import os
import numpy as np


class TaskSampler():
    def __init__(
        self,
        dataset,
        n_way: int,
        k_shot: int,
        k_query : int,
        mode_test = False,
    ):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.n_tasks = 0
        self.batch_size = self.n_way * (k_shot + k_query)
        self.items_per_label: Dict[int, List[int]] = {}
        self.mode_test = mode_test

        for ind, (_, label) in enumerate(dataset):
          if label in self.items_per_label:
            self.items_per_label[label].append(ind)
          else:
            self.items_per_label[label] = [ind]

        self.num_classes = len(self.items_per_label)
        self.batch_per_class = [0]*self.num_classes
        for c in range(self.num_classes):
            self.batch_per_class[c] = len(self.items_per_label[c]) // (self.k_shot + self.k_query)

    def __len__(self) -> int:
        return self.n_tasks

    def __iter__(self) -> Iterator[List[int]]:
        if(self.mode_test):
            random.seed(0)
        count_tasks = 0
        remain_class = [c for c, value in enumerate(self.batch_per_class) if value > 0]
        while len(remain_class) >= self.n_way:
            samples = []
            batch_per_class = list(filter(lambda x: x != 0, self.batch_per_class))
            if(self.n_way == 2):
                min_index = self.batch_per_class.index(min(batch_per_class))
                self.batch_per_class[min_index] -= 1
                max_index = self.batch_per_class.index(max(batch_per_class))
                self.batch_per_class[min_index] += 1
                class_choice = [min_index, max_index]
            else:
                class_choice = random.sample(remain_class, self.n_way)

            for c in class_choice:
                _samples = random.sample(self.items_per_label[c], self.k_shot + self.k_query)
                samples += _samples
                self.items_per_label[c] = list(set(self.items_per_label[c]).difference(_samples))
                self.batch_per_class[c] -= 1
                if(self.batch_per_class[c] == 0):
                    remain_class.remove(c)
            count_tasks += 1
            yield samples
        self.n_tasks = count_tasks


    def episodic_collate_fn(self, input_data: List[Tuple[Tensor, int]]) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[int]]:

        v_features = torch.cat([i[0] for i in input_data])
        true_class_ids = list({i[1] for i in input_data})

        all_feartures = v_features.reshape((self.n_way, self.k_shot + self.k_query, -1))
        all_labels = torch.tensor([true_class_ids.index(i[1]) for i in input_data]).reshape((self.n_way, self.k_shot + self.k_query))

        support_features = all_feartures[:, : self.k_shot].reshape((self.n_way * self.k_shot, -1))
        query_features = all_feartures[:, self.k_shot :].reshape((self.n_way * self.k_query, -1))

        support_labels = all_labels[:, : self.k_shot].flatten()
        query_labels = all_labels[:, self.k_shot :].flatten()

        return (
            support_features,
            support_labels,
            query_features,
            query_labels,
            true_class_ids,
        )

def get_dataloader(n_way, k_shot, k_query, mode_test, path):
    if path[-1] != "/":
        path += "/"
    for f in os.listdir(path):
        if(f[0] == 'X'):
          X = np.load(path+f)
        else:
          y = np.load(path+f)
    dataset = tuple(zip(torch.tensor(X), y))

    sampler = TaskSampler(
        dataset, n_way, k_shot, k_query, mode_test
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=1,
        pin_memory=True,
        collate_fn=sampler.episodic_collate_fn,
    )
    return loader
