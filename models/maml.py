import torch
from torch import nn, optim
from torch.nn import functional as F
from copy import deepcopy
from typing import List
from tqdm import tqdm
import numpy as np

class Learner_MAML(nn.Module):
    def __init__(self, config):
        if config:
            super(Learner_MAML, self).__init__()
            self.vars = nn.ParameterList()
            self.config = config
        else:
            raise NotImplementedError

        for i, (name, param) in enumerate(self.config):
            if name == "linear":
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name in ["relu", "sigmoid", 'flatten']:
                continue
            else:
                raise NotImplementedError

    def show_arch(self):
        info = ""

        for name, param in self.config:
            if name == "linear":
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'
            elif name in ['relu', 'sigmoid', 'flatten']:
                tmp = name + ":" + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info

    def forward(self, x, vars = None):
        if vars is None:
            vars = self.vars

        idx = 0

        for name, param in self.config:
            if name == "linear":
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
            elif name == 'flatten':
                x = x.view(x.size(0), -1)
            elif name == 'relu':
                x = F.relu(x, inplace=param[0])
            else:
                raise NotImplementedError

        assert idx == len(vars)
        return x

    def parameters(self):
        return self.vars

    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

class MAML(nn.Module):
    def __init__(self, args):
        super(MAML, self).__init__()

        self.num_inner_steps = args['num_inner_steps']
        self.lr_inner = args['lr_inner']
        self.lr_meta = args['lr_meta']
        self.config = [('flatten', []), 
                       ('linear', [512, 768]), 
                       ('relu', [True]), 
                       ('linear', [512, 512]), 
                       ('relu', [True]), 
                       ('linear', [128, 512]), 
                       ('relu', [True]), 
                       ('linear', [2, 128])]
        self.backbone = Learner_MAML(self.config)
        self.meta_optim = optim.Adam(self.backbone.parameters(), lr=self.lr_meta)

    def forward(
        self,
        data_loader,
    ):
        task_num = len(data_loader)
        criterion = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim = 1)
        losses_sum = 0
        corrects_sum = 0

        for iter, (
                support_features,
                support_labels,
                query_features,
                query_labels,
                class_ids,
        ) in enumerate(data_loader):

            z_support = self.backbone.forward(support_features)
            loss = criterion(z_support, support_labels)
            grad = torch.autograd.grad(loss, self.backbone.parameters(), allow_unused=True)
            # theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.lr_inner * p[0], zip(grad, self.backbone.parameters())))

            for step in range(1, self.num_inner_steps):
                z_support = self.backbone.forward(support_features, fast_weights)
                loss = criterion(z_support, support_labels)
                grad = torch.autograd.grad(loss, fast_weights, allow_unused=True)
                fast_weights = list(map(lambda p: p[1] - self.lr_inner * p[0], zip(grad, fast_weights)))

            z_query = self.backbone(query_features, fast_weights)
            loss_q = F.cross_entropy(z_query, query_labels)
            losses_sum += loss_q

            with torch.no_grad():
                pred_q = softmax(z_query).argmax(dim=1)
                correct = torch.eq(pred_q, query_labels).sum().item()
                corrects_sum += correct

        loss_q = losses_sum/task_num

        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()

        accs = corrects_sum / (query_labels.size()[0] * task_num)
        return accs, loss_q.detach().numpy()
    
    def train(self, train_loader, num_task):
        def sliding_average(value_list: List[float], window: int) -> float:
            if len(value_list) == 0:
                raise ValueError("Cannot perform sliding average on an empty list.")
            return np.asarray(value_list[-window:]).mean()

        log_update_frequency = 5
        all_loss = []
        all_acc = []

        with tqdm(enumerate(range(num_task)), total=num_task) as tqdm_train:
            for episode_index, i in tqdm_train:
                acc, loss = self.forward(train_loader)
                all_loss.append(loss)
                all_acc.append(acc)
                if episode_index % log_update_frequency == 0:
                    tqdm_train.set_postfix(loss=sliding_average(all_loss, log_update_frequency), 
                                        acc=sliding_average(all_acc, log_update_frequency))
    def test(
        self,
        data_loader,
    ):
        task_num = len(data_loader)
        criterion = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim = 1)
        corrects_sum = 0

        _backbone = deepcopy(self.backbone)

        for iter, (
                support_features,
                support_labels,
                query_features,
                query_labels,
                class_ids,
        ) in enumerate(data_loader):

            z_support = _backbone.forward(support_features)
            loss = criterion(z_support, support_labels)
            grad = torch.autograd.grad(loss, _backbone.parameters(), allow_unused=True)
            fast_weights = list(map(lambda p: p[1] - self.lr_inner * p[0], zip(grad, _backbone.parameters())))

            for step in range(1, self.num_inner_steps):
                z_support = _backbone.forward(support_features, fast_weights)
                loss = criterion(z_support, support_labels)
                grad = torch.autograd.grad(loss, fast_weights, allow_unused=True)
                fast_weights = list(map(lambda p: p[1] - self.lr_inner * p[0], zip(grad, fast_weights)))

            z_query = _backbone(query_features, fast_weights)

            with torch.no_grad():
                pred_q = softmax(z_query).argmax(dim=1)
                correct = torch.eq(pred_q, query_labels).sum().item()
                corrects_sum += correct

        del _backbone

        accs = corrects_sum / (query_labels.size()[0] * task_num)
        print(
            f"Model tested on {len(data_loader)} tasks. Accuracy: {(accs):.4f}%"
        )
        return accs