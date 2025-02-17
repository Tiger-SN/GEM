import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Gconv(nn.Module):
    """
    (Intra) graph convolution operation, with single convolutional layer
    """
    def __init__(self, in_features, out_features):
        super(Gconv, self).__init__()
        print('---in_features ',in_features)
        self.num_inputs = in_features
        self.num_outputs = out_features
        self.a_fc = nn.Linear(self.num_inputs, self.num_outputs)
        self.u_fc = nn.Linear(self.num_inputs, self.num_outputs)

    def forward(self, A, x, norm=True):
        if norm is True:
            A = F.normalize(A, p=1, dim=-2)
        #print('-------gconv------')
        #print('---A shape:', A.shape)
        #print('---x shape:', x.shape)
        # ---A shape: torch.Size([2, 20, 20])
        # ---x shape: torch.Size([2, 20, 1024])
        #x = x.view(A.shape[0],-1)
        #print('---flatten x shape:',x.shape)
        ax = self.a_fc(x)
        ux = self.u_fc(x)
        #print('--ux shape:',ux.shape)
        x = torch.bmm(A, F.relu(ax)) + F.relu(ux) # has size (bs, N, num_outputs)

        return x

class Siamese_Gconv(nn.Module):
    """
    Perform graph convolution on two input graphs (g1, g2)
    """
    def __init__(self, in_features, num_features):
        super(Siamese_Gconv, self).__init__()
        self.gconv = Gconv(in_features, num_features)

    def forward(self, g1, g2):
        emb1 = self.gconv(*g1)
        emb2 = self.gconv(*g2)
        # embx are tensors of size (bs, N, num_features)
        return emb1, emb2
