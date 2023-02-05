import torch
from torch import nn
from torch.nn import functional as F
from typing import TypeVar, Tuple, List

Tensor = TypeVar('torch.tensor')

class VAE(nn.Module):
    def __init__(self, in_features:int, latent_dim: int, layer_size: List = None, **kwargs) -> None:
        super(VAE, self).__init__()
        self._latent_dim = latent_dim
        self._in_features = in_features

        if layer_size is None:
            layer_size = [32, 32, 64]
        self._layer_size = layer_size
        
        _encoder_layers, _encoder_out_channels = self._coder_initlayers(self._in_features)
        self.encoder = nn.Sequential(*_encoder_layers)

        self.mu = nn.Linear(_encoder_out_channels, self._latent_dim)
        self.var = nn.Linear(_encoder_out_channels, self._latent_dim)

        _decoder_layers, _decoder_out_channels = self._coder_initlayers(self._latent_dim, is_encoder = False)
        self.decoder = nn.Sequential(*_decoder_layers)


        self.f = nn.Sequential(
            nn.Linear(_decoder_out_channels, _decoder_out_channels),
            nn.LeakyReLU(),
            nn.Linear(_decoder_out_channels, in_features),
        )

    def _coder_initlayers(self, in_features:int, is_encoder = True) -> Tuple[nn.Module]:
        layer_size: List = self._layer_size.copy()

        if is_encoder:
            layer_size.reverse()

        coder_layers = []
        previous = in_features

        for output in layer_size:

            coder_layers.append(nn.Linear(previous, output))
            coder_layers.append(nn.LeakyReLU())

            previous = output

        return coder_layers, output

    def _sample_from_normal(self, mu: Tensor, var: Tensor) -> Tensor:
        return torch.randn_like(mu) * torch.exp(.5 * var) + mu

    def forward(self, input: Tensor, **kwargs) -> Tuple[Tensor]:
        input = torch.flatten(input, start_dim = 1)
        encoded = self.encoder(input)
        mu, logvar = self.mu(encoded), self.var(encoded)
        eps = self._sample_from_normal(mu, logvar)
        decoded = self.decoder(eps)

        return self.f(decoded), mu, logvar

    def loss_function(self, input: Tensor, pred:Tensor, mu:Tensor, var:Tensor, dataset_size: int, **kwargs) -> dict:
        input = torch.flatten(input, start_dim = 1)
        L_recon = F.mse_loss(input, pred)

        L_reg = torch.sum(var.exp() + mu ** 2 - 1 - var, dim = 1)
        L_reg = input.size()[0] / dataset_size * torch.mean(.5 * L_reg, dim =0)
        L = L_recon + L_reg

        return {'L': L, 'L_reg': L_reg, 'L_recon': L_recon}