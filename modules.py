from math import sqrt, log
import torch as pt
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.distributions import Uniform

class HardSigmoid(nn.Module):
  def __init__(self, inplace=False):
    super().__init__()
    self.inplace = inplace

  def forward(self, x):
    if self.inplace:
      return F.hardtanh_(x, min_val=0.0, max_val=1.0)
    return F.hardtanh(x, min_val=0.0, max_val=1.0)

class SparseLinear(nn.Module):
  def __init__(self, in_features, out_features, beta=2/3, gamma=-0.1, zeta=1.1):
    super().__init__()
    self.W = nn.Parameter(pt.Tensor(in_features, out_features))
    self.b = nn.Parameter(pt.Tensor(out_features))
    self.log_alpha = nn.Parameter(pt.Tensor(in_features, out_features))
    self.beta = beta
    self.gamma = gamma
    self.zeta = zeta
    self.g = HardSigmoid()
    self.reset_parameters()
  
  def reset_parameters(self):
    init.kaiming_uniform_(self.W, a=sqrt(5))
    init.zeros_(self.b)
    init.normal_(self.log_alpha, mean=0.0, std=0.01)
  
  def forward(self, x):
    if self.training:
      # Eq. 10: Sample from the stretched binary concrete distribution.
      u = pt.rand_like(self.log_alpha)
      s = pt.sigmoid((pt.log(u) - pt.log(1 - u) + self.log_alpha) / self.beta)
      s = s * (self.zeta - self.gamma) + self.gamma
      # Eq. 11: Apply hard-sigmoid on the random sample.
      z = self.g(s)
      # Eq. 12: Compute the L0 complexity loss by evaluating the CDF of `s`.
      L_C = pt.sum(pt.sigmoid(self.log_alpha - self.beta * log(-self.gamma / self.zeta)))
      return x @ (self.W * z) + self.b, L_C

    # Eq. 13: Use the following estimator for the final parameters at test time.
    z = self.g(pt.sigmoid(self.log_alpha) * (self.zeta - self.gamma) + self.gamma)
    return x @ (self.W * z) + self.b, None
