import math

import torch
from torch import nn
from torch.nn import functional as F


def gaussian_log_prob(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    variance = sigma.pow(2)
    return (-0.5 * math.log(2.0 * math.pi) - torch.log(sigma) - (x - mu).pow(2) / (2.0 * variance)).sum()


def scale_mixture_log_prior(
    w: torch.Tensor,
    pi: float = 0.5,
    sigma1: float = 1.0,
    sigma2: float = 0.002,
    eps: float = 1e-8,
) -> torch.Tensor:
    dist1 = torch.distributions.Normal(0.0, sigma1)
    dist2 = torch.distributions.Normal(0.0, sigma2)
    prob1 = torch.exp(dist1.log_prob(w))
    prob2 = torch.exp(dist2.log_prob(w))
    mixture_prob = pi * prob1 + (1.0 - pi) * prob2
    return torch.log(mixture_prob + eps).sum()


class BayesianLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_pi: float = 0.5,
        prior_sigma1: float = 1.0,
        prior_sigma2: float = 0.002,
        mu_init_std: float = 0.1,
        rho_init_mean: float = -5.0,
        rho_init_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_pi = prior_pi
        self.prior_sigma1 = prior_sigma1
        self.prior_sigma2 = prior_sigma2

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))

        self.mu_init_std = mu_init_std
        self.rho_init_mean = rho_init_mean
        self.rho_init_std = rho_init_std
        self.reset_parameters()

        self.log_q = torch.tensor(0.0)
        self.log_p = torch.tensor(0.0)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight_mu, mean=0.0, std=self.mu_init_std)
        nn.init.normal_(self.bias_mu, mean=0.0, std=self.mu_init_std)
        nn.init.normal_(self.weight_rho, mean=self.rho_init_mean, std=self.rho_init_std)
        nn.init.normal_(self.bias_rho, mean=self.rho_init_mean, std=self.rho_init_std)

    def _sample_parameter(self, mu: torch.Tensor, rho: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = F.softplus(rho)
        epsilon = torch.randn_like(sigma)
        sample = mu + sigma * epsilon
        return sample, sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight, weight_sigma = self._sample_parameter(self.weight_mu, self.weight_rho)
        bias, bias_sigma = self._sample_parameter(self.bias_mu, self.bias_rho)

        self.log_q = gaussian_log_prob(weight, self.weight_mu, weight_sigma)
        self.log_q = self.log_q + gaussian_log_prob(bias, self.bias_mu, bias_sigma)

        self.log_p = scale_mixture_log_prior(
            weight,
            pi=self.prior_pi,
            sigma1=self.prior_sigma1,
            sigma2=self.prior_sigma2,
        )
        self.log_p = self.log_p + scale_mixture_log_prior(
            bias,
            pi=self.prior_pi,
            sigma1=self.prior_sigma1,
            sigma2=self.prior_sigma2,
        )

        return F.linear(x, weight, bias)
