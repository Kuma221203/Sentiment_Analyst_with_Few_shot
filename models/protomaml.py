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

class ProtoMAML(nn.Module):
    def __init__(self, args):
        super(ProtoMAML, self).__init__()
        self.backbone = Backbone()
        self.lr_meta = args['lr_meta']
        self.lr_output = args['lr_output']
        self.lr_inner = args['lr_inner']
        self.num_inner_steps = args['num_inner_steps']
        self.meta_optim = optim.Adam(self.backbone.parameters(), lr=self.lr_meta)
        self.meta_scheduler = None

    def calculate_prototypes(self, support_features, support_labels):
        n_way = len(torch.unique(support_labels))
        prototypes = torch.cat(
                        [
                            support_features[torch.nonzero(support_labels == label)].mean(0)
                            for label in range(n_way)
                        ]
                    )
        return prototypes

    def run_model(self, model, output_weight, output_bias, support_features, support_labels):
        feats = model.forward(support_features)
        preds = F.linear(feats, output_weight, output_bias)
        loss = F.cross_entropy(preds, support_labels)
        acc = (preds.argmax(dim=1) == support_labels).float()
        return loss, preds, acc

    def adapt_few_shot(self, support_features, support_labels):
        z_proto = self.backbone.forward(support_features)
        prototypes = self.calculate_prototypes(z_proto, support_labels)
        _backbone = deepcopy(self.backbone)
        _backbone.train()
        inner_optim = torch.optim.SGD(_backbone.parameters(), lr=self.lr_inner)
        inner_optim.zero_grad()
        init_weight = 2 * prototypes
        init_bias = -torch.norm(prototypes, dim=1)**2
        output_weight = init_weight.detach().requires_grad_()
        output_bias = init_bias.detach().requires_grad_()

        for _ in range(self.num_inner_steps):
            loss, _, _ = self.run_model(_backbone, output_weight, output_bias, support_features, support_labels)
            loss.backward()
            inner_optim.step()

            with torch.no_grad():
                output_weight.copy_(output_weight - self.lr_output * output_weight.grad)
                output_bias.copy_(output_bias - self.lr_output * output_bias.grad)

            inner_optim.zero_grad()
            output_weight.grad.fill_(0)
            output_bias.grad.fill_(0)

        output_weight = (output_weight - init_weight).detach() + init_weight
        output_bias = (output_bias - init_bias).detach() + init_bias

        return _backbone, output_weight, output_bias

    def outer_loop(self, data_loader, mode = "train", episode_index = 0):
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
            model, output_weight, output_bias = self.adapt_few_shot(support_features, support_labels)
            loss, z_query, acc = self.run_model(model, output_weight, output_bias, query_features, query_labels)

            if mode == "train":
                loss.backward()
                for p_global, p_local in zip(self.backbone.parameters(), model.parameters()):
                    p_global.grad += p_local.grad

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
        def sliding_average(value_list: List[float], window: int) -> float:
            if len(value_list) == 0:
                raise ValueError("Cannot perform sliding average on an empty list.")
            return np.asarray(value_list[-window:]).mean()

        log_update_frequency = 2
        all_loss = []
        all_acc = []

        self.meta_scheduler = StepLR(self.meta_optim, epochs//2, 0.1)
        with tqdm(enumerate(range(epochs)), total=epochs) as tqdm_train:
            for episode_index, i in tqdm_train:
                train_loader = get_dataloader(n_way, k_shot, k_query, False, path_to_data)
                loss, acc, _, _ = self.outer_loop(train_loader, mode = "train", episode_index = episode_index)
                all_loss.append(loss)
                all_acc.append(acc)
                if episode_index % log_update_frequency == 0:
                    tqdm_train.set_postfix(loss=sliding_average(all_loss, log_update_frequency),
                                            acc=sliding_average(all_acc, log_update_frequency),
                                            lr=self.meta_scheduler.get_last_lr()[0])

    def test(self, n_way, k_shot, k_query, path_to_data):
        test_loader = get_dataloader(n_way, k_shot, k_query, True, path_to_data)
        _, _, y_true, y_predict = self.outer_loop(test_loader, mode="test")
        print("\n", classification_report(y_true, y_predict, target_names=["negative", "neutral", "positive"]))
        return y_true, y_predict