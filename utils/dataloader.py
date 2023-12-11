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
        RSML: bool,
    ):

        self.dataset = dataset
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks
        self.RSML = RSML
        self.items_per_label: Dict[int, List[int]] = {}

        for ind, (_, label) in enumerate(dataset):
          if label in self.items_per_label:
            self.items_per_label[label].append(ind)
          else:
            self.items_per_label[label] = [ind]

    def __len__(self) -> int:
        return self.n_tasks

    # def __iter__(self) -> Iterator[List[int]]:
    #     for _ in range(self.n_tasks):
    #         yield torch.cat(
    #             [
    #                 torch.tensor(
    #                     random.sample(
    #                         self.items_per_label[label] , (random.randint(0, self.n_shot + self.n_query) if self.RSML else self.n_shot + self.n_query)
    #                     )
    #                 )
    #                 for label in random.sample(range(len(self.items_per_label)), self.n_way)
    #             ]
    #         ).tolist()
    
    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.n_tasks):
            sample = []
            max_ele = self.n_shot + self.n_query
            for label in random.sample(range(len(self.items_per_label)), self.n_way):
                num_ele = (random.randint(1, max_ele) if self.RSML else max_ele) # random shot meta learning
                ele = random.sample(self.items_per_label[label], num_ele)
                sample += [*ele, *random.choices(ele, k=max(0, max_ele - num_ele))] #random over sampling
            yield sample


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

def get_dataloader(n_way, n_shot, n_query, n_tasks, path, RSML = False):

    for f in os.listdir(path):
        if(f[0] == 'X'):
          X = np.load(path+f)
        else:
          y = np.load(path+f)
    dataset = tuple(zip(torch.tensor(X), y))

    sampler = TaskSampler(
        dataset, n_way, n_shot, n_query, n_tasks, RSML,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=1,
        pin_memory=True,
        collate_fn=sampler.episodic_collate_fn,
    )
    return loader