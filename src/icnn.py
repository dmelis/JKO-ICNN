#https://github.com/locuslab/icnn
#https://arxiv.org/pdf/1609.07152.pdf
import pdb
import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init, Module


MNIST_FLAT_DIM = 28*28


class ISCNN(nn.Module):
    def __init__(self, icnn, strength=1e-2):
        super(ISCNN, self).__init__()
        self.icnn = icnn
        self.strength=strength
        #self.strength = torch.nn.Parameter(torch.log(torch.exp(torch.ones(1)) - 1), requires_grad=True)
        self.output_dim = icnn.output_dim
        self.input_dim  = icnn.input_dim
    def forward(self, x):
        #pdb.set_trace()
        #return self.icnn(x) + 0*self.strength*0.5*torch.norm(x.view(x.shape[0],-1), dim=1)**2
        return self.icnn(x) + self.strength*(x.view(x.shape[0], -1) ** 2).sum(1, keepdim=True) / 2

class LinearFConvex(Module):
    """
    Applies a fully input-convex linear transformation (eq 2 in [1]) on the incoming
    data (z,y), where z is the input from the previous layer, and y is the
    original input.

    If only no prev_features is provided (i.e., instantiated with only two params),
    this assumes this is the first layer, and will expect only one input.

    Args:
        input_features: size of original input to network (y=z_0)
        in_features: size of each input sample (z_i)
        out_features: size of each output sample (z_{i+1})
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight_z: the learnable weights of the module of shape
        weight_y
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = LinearFConvex(100, 20, 80)
        >>> y = torch.rand(128, 100)
        >>> z = torch.rand(128, 20)
        >>> output = m(y,z)
        >>> print(output.shape)
        torch.Size([128, 80])

        >>> # As a first layer
        >>> m = LinearFConvex(100, None, 20)
        >>> output = m(y)
        >>> print(output.shape)
        torch.Size([128, 80])


    [1] https://arxiv.org/pdf/1609.07152.pdf


    """
    __constants__ = ['bias', 'input_features', 'out_features','prev_features']

    def __init__(self, input_features, prev_features, out_features, bias=True):#, return_y=True):
        super(LinearFConvex, self).__init__()
        self.input_features = input_features
        self.out_features = out_features
        self.prev_features = prev_features
        self.weight_y = Parameter(torch.Tensor(out_features, input_features))
        if self.prev_features is None:
            # First layer
            self.register_parameter('weight_z', None)
        else:
            self.weight_z = Parameter(torch.Tensor(out_features, prev_features))
        #self.register_parameter('weight_z', self.weight_z)
        #self.register_parameter('weight_y', self.weight_y)
        if bias:
            self.bias     = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        eps = 1e-18
#        for weight in [w for w in [self.weight_z, self.weight_y] if w is not None]:
        for weight in [w for w in [self.weight_z, self.weight_y] if w is not None]:
            init.kaiming_uniform_(weight, a=math.sqrt(5))
        if self.weight_z is not None:
            self.weight_z.data.clamp_(eps, None) # Make non-negative!
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_y) # TODO: This is currently only using one set of weights
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input_y, input_z=None):
        # if type(input_y) is tuple: # Hacky. I can't get Sequential to pass output tuples as two args
        #     input_y, input_z = input_y
        Wy = F.linear(input_y, self.weight_y, self.bias)
        if input_z is None or self.weight_z is None:
            # Z won't be passed if this is the first layer, in that case
            # this behaves like a normal Linear layer
            return Wy
        else:
            Wz = F.linear(input_z, self.weight_z, None)
            return Wz + Wy

class FICNN(nn.Module):
    """ Fully-Input Convex Neural Net """
    def __init__(
            self,
            input_dim=MNIST_FLAT_DIM,
            hidden_dims=[100,200,300],
            output_dim=1,
            dropout=0.5,
            nonlin='relu'
    ):
        super(FICNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden = nn.ModuleList()
        self.hidden.append(LinearFConvex(input_dim, None, hidden_dims[0]))
        for k in range(len(hidden_dims)-1):
            self.hidden.append(LinearFConvex(input_dim,hidden_dims[k],hidden_dims[k+1]))
        self.hidden.append(LinearFConvex(input_dim,hidden_dims[k+1],output_dim))
        #self.dropout = nn.Dropout(dropout)
        if nonlin == 'relu':
            self.nonlin = torch.relu
        elif nonlin == 'leaky_relu':
            self.nonlin = torch.nn.functional.leaky_relu
        elif nonlin == 'selu':
            self.nonlin = torch.selu

    def forward(self, X, **kwargs):
        X = X.reshape(-1, self.hidden[0].input_features)
        Z = None#torch.zeros_like(X)#X.clone() # iz Z0 = X0 or 0?
        for k in range(len(self.hidden)):
            Z = self.nonlin(self.hidden[k](X, Z))
            if k == 0:
                Z = torch.pow(Z, 2) # First nonlin need not be non-decreasing, just convex https://arxiv.org/pdf/1908.10962.pdf
            #else:
            #    Z = self.hidden[k](X, Z) # No Nonlin after last layer - otherwise can't produce negative outputs with ReLU
        #Z = torch.softmax(Z, dim=-1)
        return Z

class LinearPConvex(Module):
    """
    Applies a partially input-convex linear transformation (eq 3 in [1]) on the incoming
    data (z,y), where z is the input from the previous layer, and y is the
    original input.

    If only no prev_features is provided (i.e., instantiated with only two params),
    this assumes this is the first layer, and will expect only one input.

    Args:
        input_features: size of original input to network (y=z_0)
        in_features: size of each input sample (z_i)
        out_features: size of each output sample (z_{i+1})
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight_z: the learnable weights of the module of shape
        weight_y
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = LinearFConvex(100, 20, 80)
        >>> y = torch.rand(128, 100)
        >>> z = torch.rand(128, 20)
        >>> output = m(y,z)
        >>> print(output.shape)
        torch.Size([128, 80])

        >>> # As a first layer
        >>> m = LinearFConvex(100, None, 20)
        >>> output = m(y)
        >>> print(output.shape)
        torch.Size([128, 80])


    [1] https://arxiv.org/pdf/1609.07152.pdf


    """
    __constants__ = ['bias', 'input_features', 'out_features','prev_features']

    def __init__(self, yin_dim, zin_dim, uin_dim, out_dim, bias=True):#, return_y=True):
        super(LinearPConvex, self).__init__()
        self.yin_dim = yin_dim # dim of initial y-part of input
        self.uin_dim = uin_dim # previous dim in "x-path"
        self.zin_dim = zin_dim # previoys dim in "y-path"
        self.out_dim = out_dim


        self.weight_zu = Parameter(torch.Tensor(zin_dim,uin_dim))
        self.bias_z    = Parameter(torch.Tensor(zin_dim))
        self.weight_yu = Parameter(torch.Tensor(yin_dim,uin_dim))
        self.bias_y    = Parameter(torch.Tensor(yin_dim))

        self.weight_y  = Parameter(torch.Tensor(out_dim,yin_dim))
        self.weight_z  = Parameter(torch.Tensor(out_dim,zin_dim))
        self.weight_u  = Parameter(torch.Tensor(out_dim,uin_dim))
        self.bias_u    = Parameter(torch.Tensor(out_dim))

        self.weights = [self.weight_zu, self.weight_yu, self.weight_u, self.weight_y]
        self.biases  = [self.bias_z,    self.bias_y,    self.bias_u]

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weights:
            init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight_z.data.clamp_(0) # Only this ones has to be non-negative!
        for i,bias in enumerate(self.biases):
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights[i]) # TODO: This is currently only using one set of weights
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(bias, -bound, bound)

    def forward(self, y, z, u):
        Wzu = F.linear(u, self.weight_zu, self.bias_z)
        Wyu = F.linear(u, self.weight_yu, self.bias_y)
        Wu  = F.linear(u, self.weight_u, self.bias_u)
        Wz  = F.linear(z*F.relu(Wzu), self.weight_z, None) # Relu here is used as the x+:= max(0,x) operator
        Wy  = F.linear(y*Wyu,         self.weight_y, None)
        return Wz + Wy + Wu


class PICNN(nn.Module):
    """ Partially-Input Convex Neural Net """
    def __init__(
            self,
            x_dim = 20,
            y_dim = 20,
            u_dims=[100,200,300],
            z_dims=[150,250,350],
            nonlinu = 'relu',
            nonlinz = 'relu',
            output_dim=1,
            dropout=0.5,
    ):
        super(PICNN, self).__init__()
        self.layersu = nn.ModuleList()
        self.layersz = nn.ModuleList()
        self.u_dims = [x_dim] + u_dims + [output_dim]
        self.z_dims = [y_dim] + z_dims + [output_dim]

        if nonlinu is 'relu':
            self.nonlinu = torch.relu
        if nonlinz is 'relu':
            self.nonlinz = torch.relu

        #self.hidden.append(LinearFConvex(input_dim, None, hidden_dims[0]))
        for k in range(len(self.u_dims)-1):
            self.layersu.append(nn.Linear(self.u_dims[k], self.u_dims[k+1]))
            self.layersz.append(LinearPConvex(y_dim, self.z_dims[k], self.u_dims[k],self.z_dims[k+1]))
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, Y, **kwargs):
        # Will be convex in Y but not in X
        #X = X.reshape(-1, self.hidden[0].input_features)
        U = X.clone()
        Z = Y.clone() # CHECK: Is z_o = y or should this part be 0?
        for k in range(len(self.layersu)):
            Z = self.nonlinz(self.layersz[k](Y, Z, U))
            U = self.nonlinu(self.layersu[k](U))
        return



################################################################################
###############      METHODS PORTED FROM    ####################################
################################################################################
