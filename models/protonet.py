import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from typing import List
from tqdm import tqdm
import numpy as np

class Learner_protoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 32),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class ProtoNet(nn.Module):
    def __init__(self, args):
        super(ProtoNet, self).__init__()
        self.backbone = Learner_protoNet()
        self.criterion = nn.CrossEntropyLoss()
        self.meta_optim = optim.Adam(self.backbone.parameters(), lr=args['lr_meta'])
        self.meta_scheduler = None

    def forward(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor,
    ) -> torch.Tensor:

        z_support = self.backbone.forward(support_features)
        z_query = self.backbone.forward(query_features)

        n_way = len(torch.unique(support_labels))

        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        scores = - torch.cdist(z_query, z_proto)
        return scores

    def fit(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> float:

        self.meta_optim.zero_grad()
        classification_scores = self.forward(
            support_features, support_labels, query_features
        )

        loss = self.criterion(classification_scores, query_labels)
        loss.backward()
        self.meta_optim.step()
        self.meta_scheduler.step()

        return loss.item()
    
    def train(self, train_loader, num_task):
        def sliding_average(value_list: List[float], window: int) -> float:
            if len(value_list) == 0:
                raise ValueError("Cannot perform sliding average on an empty list.")
            return np.asarray(value_list[-window:]).mean()

        all_loss = []
        log_update_frequency = 10

        self.meta_scheduler = StepLR(self.meta_optim, num_task//4, 0.1)

        with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train:
            for episode_index, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
            ) in tqdm_train:
                loss_value = self.fit(support_images, support_labels, query_images, query_labels)
                all_loss.append(loss_value)
                if episode_index % log_update_frequency == 0:
                    tqdm_train.set_postfix(loss=sliding_average(all_loss, log_update_frequency),
                                           lr=self.meta_scheduler.get_last_lr()[0])
    
    def test(self, test_loader):
        def evaluate_on_one_task(
            support_features: torch.Tensor,
            support_labels: torch.Tensor,
            query_features: torch.Tensor,
            query_labels: torch.Tensor,
        ) -> [int, int]:
            """
            Returns the number of correct predictions of query labels, and the total number of predictions.
            """
            return (
                torch.max(
                    self.forward(support_features, support_labels, query_features)
                    .detach()
                    .data,
                    1,
                )[1]
                == query_labels
            ).sum().item(), len(query_labels)

        total_predictions = 0
        correct_predictions = 0

        with torch.no_grad():
            for episode_index, (
                support_features,
                support_labels,
                query_features,
                query_labels,
                class_ids,
            ) in tqdm(enumerate(test_loader), total=len(test_loader)):

                correct, total = evaluate_on_one_task(
                    support_features, support_labels, query_features, query_labels
                )

                total_predictions += total
                correct_predictions += correct

        print(
            f"Model tested on {len(test_loader)} tasks. Accuracy: {(100 * correct_predictions/total_predictions):.2f}%"
        )