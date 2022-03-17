import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .common import ModelInterface
from torch.cuda.amp import autocast
import math
from torchreid.losses import AngleSimpleLinear
from sklearn.decomposition import PCA

def gen_A(num_classes, t, rho, smoothing, adj_file):
    print(f"ACTUAL MATRIX PARAMS: t: {t}, rho: {rho}, smoothing: {smoothing}")
    _adj = np.load(adj_file)
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    if rho != 0.0:
        _adj = _adj * rho / (_adj.sum(0, keepdims=True) + 1e-6)
        if smoothing == 'full':
            _adj = _adj + np.identity(num_classes, np.int)
        elif smoothing == 'formula':
            _adj = _adj + np.identity(num_classes, np.int) * (1-rho)
    return _adj


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


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


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout=0.0, alpha=0.2, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime


class Image_GCNN(ModelInterface):
    def __init__(self, backbone, word_matrix, in_channel=300, adj_matrix=None, num_classes=80,
                 hidden_dim_scale=1., emb_dim_scale=1., gcn_layers=2, gcn_pooling_type='max',
                 use_last_sigmoid=True, layer_type='gcn', **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        hidden_dim = int(self.backbone.num_features / hidden_dim_scale)
        embedding_dim = int(self.backbone.num_features / emb_dim_scale)
        print(f"ACTUAL GCN DIMS: hidden_dim: {hidden_dim}, embedding_dim: {embedding_dim}")
        if layer_type == 'gcn':
            self.layer_block = GraphConvolution
        elif layer_type == 'gan':
            self.layer_block = GraphAttentionLayer
        else:
            self.layer_block = SpGraphAttentionLayer

        self.num_classes = num_classes
        if gcn_layers == 1:
            self.gc1 = self.layer_block(in_channel, embedding_dim)
        self.pooling = gcn_pooling_type
        self.gcn_layers = gcn_layers
        self.use_last_sigmoid = use_last_sigmoid
        if gcn_layers == 2:
            self.gc1 = self.layer_block(in_channel, hidden_dim)
            self.gc2 = self.layer_block(hidden_dim, embedding_dim)
        elif gcn_layers == 3:
            self.gc1 = self.layer_block(in_channel, hidden_dim)
            self.gc2 = self.layer_block(hidden_dim, hidden_dim)
            self.gc3 = self.layer_block(hidden_dim, embedding_dim)

        self.relu = nn.LeakyReLU(0.2)
        # nmf = PCA(n_components=64)
        # word_matrix = nmf.fit_transform(np.transpose(word_matrix))
        # self.inp = nn.Parameter(torch.from_numpy(np.transpose(word_matrix)).float())
        self.inp = nn.Parameter(torch.from_numpy(word_matrix).float())
        self.A = nn.Parameter(torch.from_numpy(adj_matrix).float())
        # self.proj_embed = nn.Linear(self.backbone.num_features, self.num_classes * embedding_dim, bias=False)
        # self.proj_embed.weight = torch.nn.init.xavier_normal_(self.proj_embed.weight)
        # self.prototypes = nn.Parameter(torch.Tensor(self.num_classes, embedding_dim))
        # self.prototypes.data.normal_().renorm_(2, 1, 1e-5).mul_(1e5)
        self.dropout = self.backbone.dropout
        # self.proj_linear = nn.Linear(embedding_dim, self.backbone.num_features)
        if self.loss == "am_binary":
            self.head = AngleSimpleLinear(self.backbone.num_features, self.num_classes)
        else:
            self.head = nn.Linear(self.backbone.num_features, self.num_classes)
        # self.counting_head = nn.Linear(self.backbone.num_features, 1)

    def forward(self, image, return_embedings=False):
        with autocast(enabled=self.mix_precision):
            spat_features = self.backbone(image, return_featuremaps=True)

            adj = self.gen_adj(self.A).detach()
            x = self.gc1(self.inp, adj)
            x = self.relu(x)
            if self.gcn_layers > 1:
                x = self.gc2(x, adj)
                x = self.relu(x)
                if self.gcn_layers == 3:
                    x = self.gc3(x, adj)
                    x = self.relu(x)
            if self.pooling == 'max':
                weights = x.max(dim=0)[0]
            elif self.pooling == 'avg':
                weights = x.mean(dim=0)
            else:
                weights = x.mean(dim=0) + x.max(dim=0)[0]
            # weights = self.proj_linear(weights)
            if self.use_last_sigmoid:
                weights = torch.sigmoid(weights)

            weighted_cam = weights.view(1, -1, 1, 1) * spat_features
            glob_features = self.backbone.glob_feature_vector(weighted_cam, self.backbone.pooling_type, reduce_dims=False)
            # glob_features = self.dropout(glob_features)
            # count_num = self.counting_head(glob_features.view(image.size(0), -1))
            # projection head
            ###
            # embedings = self.proj_embed(glob_features.view(image.size(0), -1))
            # embedings = embedings.reshape(image.size(0), self.num_classes, -1)
            # logits = F.cosine_similarity(embedings, x, dim=2)
            # logits = logits.clamp(-1, 1)
            ###
            logits = self.head(glob_features.view(glob_features.size(0), -1))
            # logits = [logits, count_num]

            if self.similarity_adjustment:
                logits = self.sym_adjust(logits, self.similarity_adjustment)


            if not self.training:
                return [logits]

            elif self.loss in ['asl', 'bce', 'am_binary']:
                if return_embedings:
                    embedings = None
                    return tuple([logits]), x, embedings
                else:
                    out_data = [logits]

            else:
                raise KeyError("Unsupported loss: {}".format(self.loss))

            return tuple(out_data)

    @staticmethod
    def gen_adj(A):
        # model = NMF(n_components=40, n_samples=40)
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D)
        # adj = F.interpolate(adj.reshape(1,1,500,500), size=(64,64), mode='bilinear')
        # W = model.fit_transform(adj.detach().numpy())
        # print(W, W.shape)
        # return adj.reshape(64,64)
        return adj

    def get_config_optim(self, lrs):
        parameters = [
            {'params': self.head.named_parameters()},
            # {'params': self.proj_linear.named_parameters()},
            # {'params': [('proto.weights', self.prototypes), ]},
            # {'params': self.counting_head.named_parameters()},
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
                    thau = 0.4, rho_gcn=0.25, smoothing='full', pretrain=False, **kwargs):
    adj_matrix = gen_A(num_classes, thau, rho_gcn, smoothing, adj_file)
    word_matrix = np.load(word_matrix_path, allow_pickle=True)
    model = Image_GCNN(
        backbone=backbone,
        word_matrix=word_matrix,
        adj_matrix=adj_matrix,
        pretrain=pretrain,
        in_channel=word_emb_size,
        num_classes=num_classes,
        **kwargs
    )
    return model
