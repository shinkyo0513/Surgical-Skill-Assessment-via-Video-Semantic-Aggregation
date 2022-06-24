import torch
from torch import nn
# import torch.nn.functional as F
# from torch.nn.parameter import Parameter
import math


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # input: B x N_vec x C_in
        # adj:   B x N_vec x N_vec
        # support = torch.mm(input, self.weight)  # B x N_vec x C_out
        # output = torch.mm(adj, support)         # B x N_vec x C_out
        support = torch.einsum('bni,io->bno', input, self.weight)
        # print(f'support: {support.shape}')
        output = torch.einsum('bnm,bno->bmo', adj, support)
        # print(f'output: {output.shape}')
        #output = SparseMM(adj)(support)
        # output: B x N_vec x N_vec
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nmid, nout):
        super(GCN, self).__init__()
        self.nfeat = nfeat

        self.gc1 = GraphConvolution(nfeat, nmid)
        self.gc2 = GraphConvolution(nmid, nout)

        # self.norm_in = nn.BatchNorm1d(nfeat)
        # self.norm_mid = nn.BatchNorm1d(nmid)
        # self.norm_out = nn.BatchNorm1d(nout)
        self.norm_in = nn.InstanceNorm1d(nfeat)
        self.norm_mid = nn.InstanceNorm1d(nmid)
        self.norm_out = nn.InstanceNorm1d(nout)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        # x: B x N_vec x C_in
        # adj: B x N_vec x N_vec
        assert self.nfeat == x.shape[2]

        # identity = x

        # out = self.norm_in(x.transpose(1,2)).transpose(1,2).contiguous()
        # out = self.relu(out)
        # out = self.gc1(out, adj)    # B x N_vec x C_mid

        # out = self.norm_mid(out.transpose(1,2)).transpose(1,2).contiguous()
        # out = self.relu(out)
        # out = self.gc2(out, adj)    # B x N_vec x C_out

        # out += identity
        # out = self.relu(out)

        out = self.relu(self.gc1(x, adj))
        out = self.gc2(out, adj)
        
        return out

