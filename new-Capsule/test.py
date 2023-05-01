from torch import nn
import torch
import math
import torch.nn.functional as F


class EMRouting(nn.Module):
    def __init__(self, in_capsules, out_capsules, in_dim, out_dim, num_iterations=3):
        super(EMRouting, self).__init__()
        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_iterations = num_iterations
        self.epsilon = 1e-9
        self.beta_v = nn.Parameter(torch.randn(out_capsules, 1))
        self.beta_a = nn.Parameter(torch.randn(out_capsules, 1))
        self.W = nn.Parameter(torch.randn(1, in_capsules, out_capsules, out_dim, in_dim))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(2).unsqueeze(4)
        W = self.W.repeat(batch_size, 1, 1, 1, 1)
        u_hat = torch.matmul(W, x)

        # Initialize logits b to zero
        b = torch.zeros(batch_size, self.in_capsules, self.out_capsules, 1).to(x.device)

        for _ in range(self.num_iterations):
            # E-step
            c = F.softmax(b, dim=2)
            s = (c * u_hat).sum(dim=1, keepdim=True)
            v = self.squash(s)
            v_norm = v.pow(2).sum(dim=-2, keepdim=True)

            # M-step
            agreement = (u_hat * v).sum(dim=-1, keepdim=True)
            log_p = -0.5 * torch.log(2 * math.pi * self.epsilon) - 0.5 * (u_hat - v).pow(2).sum(dim=-1) / self.epsilon
            log_p = log_p - agreement - self.beta_v - self.beta_a
            b = b + log_p

        return v.squeeze(1)

    def squash(self, x):
        norm = x.pow(2).sum(dim=-2, keepdim=True).sqrt()
        x = norm / (1 + norm.pow(2)) * x
        return x
    
if __name__ == '__main__':
    em = EMRouting(in_capsules = 10, out_capsules = 2, in_dim = 4, out_dim = 4)
    v = torch.randn((2, 10, 4))
    out = em(v)
    print(out.shape)