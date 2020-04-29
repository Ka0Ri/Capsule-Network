
import torch
x = torch.rand((2, 20, 10, 10, 4))

K = 3
stride = 2
b, B, h, w, psize = x.shape
oh = ow = int((h - K + 1) / stride)
idxs = [[(h_idx + k_idx) for k_idx in range(0, K)] for h_idx in range(0, h - K + 1, stride)]
print(idxs)
x = x[:, :, idxs, :, :]
print(x.size())
x = x[:, :, :, :, idxs, :]
print(x.size())
x = x.permute(0, 1, 3, 2, 4, 5, 6).contiguous()

x = x.view(b, oh, ow, -1, psize)


def caps_em_routing(v, a_in, beta_u, beta_a, iters):
    """
    i in B: layer l, j in h*w*C: layer l + 1
    with each cell in h*w, compute on matrix B*C
    Input:
        v:         (b, h, w, B, C, P*P)
        a_in:      (b, h, w, B, 1)
    Output:
        mu:        (b, h, w, C, P*P)
        a_out:     (b, h, w, C, 1)
        'b = batch_size'
        'h = height'
        'w = width'
        'C = depth'
        'B = K*K*B'
        'psize = P*P'
    """
    eps = 1e-06
    _lambda = 1e-03
    ln_2pi = torch.cuda.FloatTensor(1).fill_(np.log(2*np.pi))
    b, h, w, B, C, psize = v.shape

    for iter_ in range(iters):
        #E step
        if(iter_ == 0):
            r = torch.cuda.FloatTensor(b, h, w, B, C).fill_(1./C) * a_in
        else:
            ln_pjh = -1. * (v - mu)**2 / (2 * sigma_sq) - torch.log(sigma_sq.sqrt()) - 0.5*ln_2pi
            a_out = a_out.view(b, h, w, 1, C)
            ln_ap = ln_pjh.sum(dim=5) + torch.log(a_out)
            r = F.softmax(ln_ap, dim=4)
            r = r * a_out
           
        #M step
        r_sum = r.sum(dim=3, keepdim=True)
        coeff = r / (r_sum + eps)
        coeff = coeff.view(b, h, w, B, C, 1)
        mu = torch.sum(coeff * v, dim=3, keepdim=True)
        sigma_sq = torch.sum(coeff * (v - mu)**2, dim=3, keepdim=True) + eps
        r_sum = r_sum.view(b, h, w, 1, C, 1)
        cost_h = (beta_u + 0.5*torch.log(sigma_sq)) * r_sum
        logit = _lambda*(beta_a - cost_h.sum(dim=5, keepdim=True))
        a_out = torch.sigmoid(logit)
        
        
    mu = mu.view(b, h, w, C, psize)
    a_out = a_out.view(b, h, w, C, 1)
  
    return mu, a_out


class glocapBlockFuzzy(nn.Module):
    def __init__(self, in_channels, cap_dim, n_classes, settings, dim=(4,4)):
        super(glocapBlockFuzzy, self).__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        n = self.n_classes * in_channels
        self.P = dim[0]
        self.m = settings['m']
        self.eps = 1e-06
        self._lambda = 1
        # self.pw = CapLayer(in_channels, n, kernel_size=(1,1),
        #         stride=(1,1), out_dim=dim, in_dim=dim, groups=in_channels)
        self.weight = nn.Parameter(torch.randn(1, n_classes, in_channels, 1, self.P, self.P), requires_grad=True)
        self.beta_a = nn.Parameter(torch.zeros(n_classes,1), requires_grad=True)
        self.n_routs = settings['n_rout']
        
    def forward(self, l, g):
        """
        Fuzzy routing
        input:
        -l: capsules at an intermediate layer (N, C_in, D, W, H)
        N: batch size, C_in: number of channels, D: capsule dim
        W, H: spatial dim
        -g: global capsule (N, C_out, D)
        output: 
        -g: updated global capsule
        -b: attention scores (N, C_out, C_in)
        -c: coefficients, attention (N, C_out, C_in)
        """
        #learnable transform
        N, C, D, W, H = l.size()
        l = l.permute(0, 1, 3, 4, 2).contiguous().view(N, C, W * H, self.P, self.P)
        V = l[:, None, :, :, :, :] @ self.weight
        V = V.view(N, self.n_classes, self.in_channels*W*H, D)
        g = g[:, :, None, :]

        for i in range(self.n_routs):
            #fuzzy coeff
            r_n = (torch.norm((V - g), dim=-1)) ** (2/(self.m - 1))
            r_d = torch.sum(1. / (r_n), dim=1, keepdim=True)
            r = (1. / (r_n * r_d)) ** (self.m)
 
            #update pose
            r_sum = r.sum(dim = 2, keepdim=True)
            #coeff = F.softmax(r, dim=1)
            coeff = r / r_sum
            coeff = coeff[:, :, :, None]
            g = torch.sum(coeff * V, dim=2, keepdim=True)

        #calcuate probability
        sigma_sq = torch.sum(torch.sum(coeff * (V - g) ** 2, dim=-1), dim=2, keepdim=True)
        a = torch.sigmoid(self._lambda*(self.beta_a - 0.5*torch.log(sigma_sq)))

        g = g.squeeze()
        a = a.squeeze()
        return a, g