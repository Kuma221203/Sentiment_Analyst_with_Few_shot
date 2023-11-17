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
        n_shot: int,
        n_query: int,
        n_tasks: int,
    ):

        self.dataset = dataset
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks
        self.items_per_label: Dict[int, List[int]] = {}

        for ind, (_, label) in enumerate(dataset):
          if label in self.items_per_label:
            self.items_per_label[label].append(ind)
          else:
            self.items_per_label[label] = [ind]

    def __len__(self) -> int:
        return self.n_tasks

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.n_tasks):
            yield torch.cat(
                [
                    torch.tensor(
                        random.sample(
                            self.items_per_label[label], self.n_shot + self.n_query
                        )
                    )
                    for label in random.sample(range(3), self.n_way)
                ]
            ).tolist()


    def episodic_collate_fn(self, input_data: List[Tuple[Tensor, int]]) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[int]]:

        v_features = torch.cat([i[0] for i in input_data])
        true_class_ids = list({i[1] for i in input_data})

        all_feartures = v_features.reshape((self.n_way, self.n_shot + self.n_query, -1))
        all_labels = torch.tensor([true_class_ids.index(i[1]) for i in input_data]).reshape((self.n_way, self.n_shot + self.n_query))

        support_features = all_feartures[:, : self.n_shot].reshape((self.n_way * self.n_shot, -1))
        query_features = all_feartures[:, self.n_shot :].reshape((self.n_way * self.n_query, -1))

        support_labels = all_labels[:, : self.n_shot].flatten()
        query_labels = all_labels[:, self.n_shot :].flatten()

        return (
            support_features,
            support_labels,
            query_features,
            query_labels,
            true_class_ids,
        )

def get_dataloader(n_way, n_shot, n_query, n_tasks, path):

    for f in os.listdir(path):
        if(f[0] == 'X'):
          X = np.load(path+f)
        else:
          y = np.load(path+f)
    dataset = tuple(zip(torch.tensor(X), y))

    sampler = TaskSampler(
        dataset, n_way = n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_tasks
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=1,
        pin_memory=True,
        collate_fn=sampler.episodic_collate_fn,
    )

    return loader
   
