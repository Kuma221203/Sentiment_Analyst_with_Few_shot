import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
from typing import List
from tqdm import tqdm
import numpy as np
from utils.dataloader import get_dataloader
from sklearn.metrics import classification_report

class Backbone(nn.Module):
    def __init__(self, n_way):
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
            nn.Linear(32, n_way),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class MAML(nn.Module):
    def __init__(self, args):
        super(MAML, self).__init__()
        self.backbone = Backbone()
        self.lr_meta = args['lr_meta']
        self.lr_inner = args['lr_inner']
        self.num_inner_steps = args['num_inner_steps']
        self.n_way = args['n_way']
        self.meta_optim = optim.Adam(self.backbone.parameters(), lr=self.lr_meta)
        self.meta_scheduler = None

    def run_model(self, model, support_features, support_labels):
        preds = model.forward(support_features)
        loss = F.cross_entropy(preds, support_labels)
        acc = (preds.argmax(dim=1) == support_labels).float()
        return loss, preds, acc

    def adapt_few_shot(self, support_features, support_labels):
        _backbone = deepcopy(self.backbone)
        inner_optim = torch.optim.SGD(_backbone.parameters(), lr=self.lr_inner)
        self.inner_scheduler = StepLR(inner_optim, self.num_inner_steps//4, 0.8)
        inner_optim.zero_grad()

        for _ in range(self.num_inner_steps):
            loss, _, _ = self.run_model(_backbone, support_features, support_labels)
            loss.backward()
            inner_optim.step()
            inner_optim.zero_grad()

        return _backbone

    def outer_loop(self, data_loader, mode = "train"):
        accs = []
        losses = []
        y_true = []
        y_predict = []
        softmax = nn.Softmax(dim = 1)
        self.backbone.zero_grad()
        for iter, (
                support_features,
                support_labels,
                query_features,
                query_labels,
                class_ids,
        ) in enumerate(data_loader):
            model= self.adapt_few_shot(support_features, support_labels)
            loss, z_query, acc = self.run_model(model, query_features, query_labels)

            if mode == "train":
                loss.backward()
                for p_global, p_local in zip(self.backbone.parameters(), model.parameters()):
                    p_global.grad = p_local.grad if p_global.grad is None else (p_global.grad + p_local.grad)

            with torch.no_grad():
                z_predict = softmax(z_query).argmax(dim=1)
                _query_labels = query_labels.clone()
                _z_predict = z_predict.clone()
                if(len(class_ids) == 2):
                    query_labels[_query_labels == 0] = min(class_ids)
                    query_labels[_query_labels == 1] = max(class_ids)
                    z_predict[_z_predict == 0] = min(class_ids)
                    z_predict[_z_predict == 1] = max(class_ids)
                y_true += query_labels.int().tolist()
                y_predict += z_predict.int().tolist()

            accs.append(acc.mean().detach())
            losses.append(loss.detach())


        if mode == "train":
            self.meta_optim.step()
            self.meta_scheduler.step()
            self.meta_optim.zero_grad()
        return float(sum(losses) / len(losses)), round(float(sum(accs) / len(accs)), 5), y_true, y_predict

    def train(self, epochs, n_way, k_shot, k_query, path_to_data):
        assert n_way == self.n_way
        def sliding_average(value_list: List[float], window: int) -> float:
            if len(value_list) == 0:
                raise ValueError("Cannot perform sliding average on an empty list.")
            return np.asarray(value_list[-window:]).mean()

        log_update_frequency = 3
        all_loss = []
        all_acc = []

        self.meta_scheduler = StepLR(self.meta_optim, epochs//2, 0.5)
        with tqdm(enumerate(range(epochs)), total=epochs) as tqdm_train:
            for episode_index, i in tqdm_train:
                train_loader = get_dataloader(n_way, k_shot, k_query, False, path_to_data)
                loss, acc, _, _ = self.outer_loop(train_loader, mode = "train")
                all_loss.append(loss)
                all_acc.append(acc)
                if episode_index % log_update_frequency == 0:
                    tqdm_train.set_postfix(loss=sliding_average(all_loss, log_update_frequency),
                                            acc=sliding_average(all_acc, log_update_frequency),
                                            lr=self.meta_scheduler.get_last_lr()[0])

    def test(self, n_way, k_shot, k_query, path_to_data):
        assert n_way == self.n_way
        self.lr_inner = 0.005
        self.num_inner_steps = 20
        test_loader = get_dataloader(n_way, k_shot, k_query, True, path_to_data)
        _, _, y_true, y_predict = self.outer_loop(test_loader, mode="test")
        print("\n", classification_report(y_true, y_predict, target_names=["negative", "neutral", "positive"]))
        return y_true, y_predict