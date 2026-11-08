"""RL token (after "RL Token", rlt.pdf): a one-token bottleneck over the frozen ACT encoder.

The ACT transformer encoder's output for an observation (n_tokens x dim_model: latent, state,
camera feature tokens, see lerobot_policy.Policy.predict_chunk_and_embedding) is projected
to d_model, a learned RL token is appended, and a small transformer encoder reads them all;
its output at the RL token is z_rl. A decoder reconstructs every input token from z_rl alone
(one learned query per token position, cross-attending to z_rl), trained with an MSE loss on
the per-dimension standardized embeddings. The online RL actor and critic then take z_rl
(plus the proprioceptive state) as their observation.

Unlike the paper's autoregressive decoder (teacher-forced on the previous tokens), the
queries never see the true tokens, so everything the reconstruction needs has to go through
z_rl.
"""

import torch
from torch import nn


class RLToken(nn.Module):

    def __init__(self, n_tokens, dim_in, d_model=256, n_heads=8, n_enc_layers=2,
                 n_dec_layers=2, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.hparams = dict(n_tokens=n_tokens, dim_in=dim_in, d_model=d_model, n_heads=n_heads,
                            n_enc_layers=n_enc_layers, n_dec_layers=n_dec_layers,
                            dim_feedforward=dim_feedforward, dropout=dropout)
        # Standardization of the inputs, set from the training data (set_normalization).
        self.register_buffer('emb_mean', torch.zeros(dim_in))
        self.register_buffer('emb_std', torch.ones(dim_in))
        self.in_proj = nn.Linear(dim_in, d_model)
        self.pos = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.rl_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        def layer(cls):
            return cls(d_model, n_heads, dim_feedforward, dropout, batch_first=True,
                       norm_first=True)

        self.encoder = nn.TransformerEncoder(layer(nn.TransformerEncoderLayer), n_enc_layers,
                                             enable_nested_tensor=False)
        self.z_norm = nn.LayerNorm(d_model)
        self.queries = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.decoder = nn.TransformerDecoder(layer(nn.TransformerDecoderLayer), n_dec_layers)
        self.out_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, dim_in))

    @property
    def z_dim(self):
        return self.hparams['d_model']

    def set_normalization(self, mean, std):
        self.emb_mean.copy_(torch.as_tensor(mean))
        self.emb_std.copy_(torch.as_tensor(std).clamp_min(1e-3))

    def normalize(self, emb):
        return (emb.float() - self.emb_mean) / self.emb_std

    def encode(self, emb):
        """(B, n_tokens, dim_in) raw ACT encoder output -> z_rl (B, d_model)."""
        x = self.in_proj(self.normalize(emb)) + self.pos
        x = torch.cat([x, self.rl_token.expand(len(x), -1, -1)], dim=1)
        return self.z_norm(self.encoder(x)[:, -1])

    def decode(self, z):
        """z_rl (B, d_model) -> reconstructed standardized embedding (B, n_tokens, dim_in)."""
        queries = self.queries.expand(len(z), -1, -1)
        return self.out_proj(self.decoder(queries, z[:, None]))

    def forward(self, emb):
        """(reconstruction loss, z_rl)."""
        z = self.encode(emb)
        loss = nn.functional.mse_loss(self.decode(z), self.normalize(emb))
        return loss, z

    def save(self, path, **extra):
        torch.save({'hparams': self.hparams, 'state_dict': self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path, map_location='cpu'):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**ckpt['hparams'])
        model.load_state_dict(ckpt['state_dict'])
        return model.eval(), ckpt
