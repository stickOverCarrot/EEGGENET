import numpy as np
import torch as th
from torch import nn
from braindecode.torch_ext.init import glorot_weight_zero_bias
from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.util import np_to_var
from abc import abstractmethod
from .utils import normalize_adj


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, inputs):
        """
        Forward pass logic
        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1., **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:
            self.weight.data = th.renorm(self.weight.data, p=2, dim=0,
                                         maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=1., **kwargs):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:
            self.weight.data = th.renorm(self.weight.data, p=2, dim=0,
                                         maxnorm=self.max_norm)
        return super(LinearWithConstraint, self).forward(x)


def _transpose_to_0312(x):
    return x.permute(0, 3, 1, 2)


def _transpose_to_0132(x):
    return x.permute(0, 1, 3, 2)


def _review(x):
    return x.contiguous().view(-1, x.size(2), x.size(3))


def _squeeze_final_output(x):
    """
    Remove empty dim at end and potentially remove empty time dim
    Do not just use squeeze as we never want to remove first dim
    :param x:
    :return:
    """
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


class GraphEmbedding(nn.Module):
    def __init__(self,
                 n_nodes,
                 input_dim,
                 adj,
                 k=1,
                 adj_learn=True):
        super(GraphEmbedding, self).__init__()
        self.__dict__.update(locals())
        del self.self
        self.xs, self.ys = th.tril_indices(self.n_nodes, self.n_nodes, offset=-1)
        node_value = adj[self.xs, self.ys]
        self.edge_weight = nn.Parameter(node_value.clone().detach(), requires_grad=self.adj_learn)

    def forward(self, x):
        edge_weight = th.zeros([self.n_nodes, self.n_nodes], device=x.device)
        edge_weight[self.xs.to(x.device), self.ys.to(x.device)] = self.edge_weight.to(x.device)
        edge_weight = edge_weight + edge_weight.T + th.eye(self.n_nodes, dtype=edge_weight.dtype, device=x.device)
        edge_weight = normalize_adj(edge_weight, mode='sym')
        x_out = [x]
        for k in range(self.k):
            x = th.matmul(edge_weight.unsqueeze(0), x)
            x_out.append(x)
        x_out = th.stack(x_out)
        return x_out


class EEGGENET(BaseModel):
    def __init__(self,
                 Adj,
                 in_chans,
                 n_classes,
                 final_conv_length='auto',
                 input_time_length=None,
                 pool_mode='mean',
                 k=2,
                 f1=8,
                 d=2,
                 f2=16,  # usually set to F1*D (?)
                 kernel_length=64,
                 third_kernel_size=(8, 4),
                 drop_prob=0.25,
                 Adj_learn=False,
                 ):
        super(EEGGENET, self).__init__()

        if final_conv_length == 'auto':
            assert input_time_length is not None

        self.__dict__.update(locals())
        del self.self

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]

        self.temporal_conv = nn.Sequential(
            Expression(_transpose_to_0312),
            Conv2dWithConstraint(in_channels=1, out_channels=self.f1,
                                 kernel_size=(1, self.kernel_length),
                                 max_norm=None,
                                 stride=1,
                                 bias=False,
                                 padding=(0, self.kernel_length // 2)
                                 ),
            nn.BatchNorm2d(self.f1, momentum=0.01, affine=True, eps=1e-3),
        )

        self.ge = nn.Sequential(
            Expression(_review),
            GraphEmbedding(self.in_chans, self.input_time_length, adj=self.Adj, k=self.k, adj_learn=self.Adj_learn),
        )

        self.spatial_conv = nn.Sequential(
            Conv2dWithConstraint(self.f1 * (self.k + 1), self.f1 * self.d * (self.k + 1), (self.in_chans, 1),
                                 max_norm=1, stride=1, bias=False,
                                 groups=self.f1 * (self.k + 1), padding=(0, 0)),
            nn.BatchNorm2d(self.f1 * self.d * (self.k + 1), momentum=0.01, affine=True,
                           eps=1e-3),
            nn.ELU(),
            pool_class(kernel_size=(1, 8), stride=(1, 8))
        )

        self.separable_conv = nn.Sequential(
            nn.Dropout(p=self.drop_prob),
            Conv2dWithConstraint(self.f1 * self.d * (self.k + 1), self.f1 * self.d * (self.k + 1), (1, 16),
                                 max_norm=None,
                                 stride=1,
                                 bias=False, groups=self.f1 * self.d * (self.k + 1),
                                 padding=(0, 8)),
            Conv2dWithConstraint(self.f1 * self.d * (self.k + 1), self.f2, (1, 1), max_norm=None, stride=1, bias=False,
                                 padding=(0, 0)),
            nn.BatchNorm2d(self.f2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            pool_class(kernel_size=(1, 8), stride=(1, 8))
        )

        out = np_to_var(np.ones((1, self.in_chans, self.input_time_length, 1), dtype=np.float32))
        out = self.forward_init(out)
        n_out_virtual_chans = out.cpu().data.numpy().shape[2]

        if self.final_conv_length == 'auto':
            n_out_time = out.cpu().data.numpy().shape[3]
            self.final_conv_length = n_out_time

        # Classifier part:
        self.cls = nn.Sequential(
            nn.Dropout(p=self.drop_prob),
            Conv2dWithConstraint(self.f2, self.n_classes,
                                 (n_out_virtual_chans, self.final_conv_length), max_norm=0.25,
                                 bias=True),
            Expression(_transpose_to_0132),
            Expression(_squeeze_final_output)
        )

        # Initialize weights of the network
        self.apply(glorot_weight_zero_bias)

    def forward_init(self, x):
        with th.no_grad():
            batch_size = x.size(0)
            x = self.temporal_conv(x)
            x = self.ge(x)
            x = x.view((self.k + 1), batch_size, -1, x.size(-2), x.size(-1))
            x = x.permute(1, 2, 0, 3, 4).contiguous().view(batch_size, -1, x.size(-2), x.size(-1))
            x = self.spatial_conv(x)
            x = self.separable_conv(x)
        return x

    def forward(self, inputs):
        return self.forward_once(inputs)

    def forward_once(self, x):
        batch_size = x.size(0)
        x = x[:, :, :, None]
        x = self.temporal_conv(x)
        x = self.ge(x)
        x = x.view((self.k + 1), batch_size, -1, x.size(-2), x.size(-1))
        x = x.permute(1, 0, 2, 3, 4).contiguous().view(batch_size, -1, x.size(-2), x.size(-1))
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        x = self.cls(x)
        return x
