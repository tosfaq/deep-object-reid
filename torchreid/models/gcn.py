import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .common import ModelInterface
from torch.cuda.amp import autocast
import math

def gen_A(num_classes, t, rho, adj_file):
    _adj = np.load(adj_file)
    # _adj = result['adj']
    # _nums = result['nums']
    # _nums = _nums[:, np.newaxis]
    # _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * rho / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, _input, adj):
        support = torch.matmul(_input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Image_GCNN(ModelInterface):
    def __init__(self, backbone, word_matrix, in_channel=300, adj_matrix=None, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.pooling = nn.MaxPool2d(14, 14)
        self.gc1 = GraphConvolution(in_channel, self.backbone.num_features // 2)
        self.gc2 = GraphConvolution(self.backbone.num_features // 2, self.backbone.num_features)
        self.relu = nn.LeakyReLU(0.2)
        self.inp = nn.Parameter(torch.from_numpy(word_matrix).float())
        self.A = nn.Parameter(torch.from_numpy(adj_matrix).float())

    def forward(self, image):
        with autocast(enabled=self.mix_precision):
            feature = self.backbone(image, return_featuremaps=True)
            glob_features = self.pooling(feature)

            adj = self.gen_adj(self.A).detach()
            x = self.gc1(self.inp, adj)
            x = self.relu(x)
            x = self.gc2(x, adj)

            x = x.transpose(0, 1)

            logits = F.normalize(glob_features.view(glob_features.shape[0], -1), p=2, dim=1).mm(F.normalize(x, p=2, dim=0))
            logits = logits.clamp(-1, 1)

            if self.similarity_adjustment:
                logits = self.sym_adjust(logits, self.similarity_adjustment)

            if not self.training:
                return [logits]

            elif self.loss in ['asl', 'bce', 'am_binary']:
                    out_data = [logits]
            else:
                raise KeyError("Unsupported loss: {}".format(self.loss))

            return tuple(out_data)

    @staticmethod
    def gen_adj(A):
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D)
        return adj

    def get_config_optim(self, lrs):
        parameters = [
            {'params': self.backbone.named_parameters()},
            {'params': self.gc1.named_parameters()},
            {'params': self.gc2.named_parameters()},
        ]
        if isinstance(lrs, list):
            assert len(lrs) == len(parameters)
            for lr, param_dict in zip(lrs, parameters):
                param_dict['lr'] = lr
        else:
            assert isinstance(lrs, float)
            for i, param_dict in enumerate(parameters):
                    param_dict['lr'] = lrs

        return parameters


def build_image_gcn(backbone, word_matrix_path, adj_file, num_classes=80, word_emb_size=300,
                    thau = 0.4, rho_gcn=0.25, pretrain=False, **kwargs):
    print(thau)
    adj_matrix = gen_A(num_classes, thau, rho_gcn, adj_file)
    word_matrix = np.load(word_matrix_path)
    model = Image_GCNN(
        backbone=backbone,
        word_matrix=word_matrix,
        adj_matrix=adj_matrix,
        pretrain=pretrain,
        in_channel=word_emb_size,
        **kwargs
    )
    return model
