import torch
import torch.nn as nn


class CameraModulation(nn.Module):
    """
    Adaptive Layer Normalization and modulation
    """

    def __init__(self, hidden_dim, camera_embed_dim, eps):
        super(CameraModulation, self).__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.gamma_beta_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(camera_embed_dim, hidden_dim * 2),
        )

    def forward(self, triplane_hidden_features, camera_features):
        """
        :param triplane_hidden_features: [batch, seq_len, dim]
        :param camera_features: [batch, camera_dim]
        :return:
        """
        gamma, beta = self.gamma_beta_mlp(camera_features).chunk(2, dim=-1)  # [batch, dim]
        x = self.norm(triplane_hidden_features)
        x = x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)  # [batch, seq_len, dim]
        return x


class Block(nn.Module):
    def __init__(self, hidden_dim, cond_dim, mod_dim, num_heads, eps,
                 attn_drop=0., attn_bias=False,
                 mlp_forward_expansion=4., mlp_dropout=0.10):
        super(Block, self).__init__()
        self.modulated_norm_1 = CameraModulation(hidden_dim, mod_dim, eps)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, kdim=cond_dim, vdim=cond_dim,
                                                dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.modulated_norm_2 = CameraModulation(hidden_dim, mod_dim, eps)
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,
                                               dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.modulated_norm_3 = CameraModulation(hidden_dim, mod_dim, eps)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_forward_expansion)),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(int(hidden_dim * mlp_forward_expansion), hidden_dim),
            nn.Dropout(mlp_dropout),
        )

    def forward(self, x, img_feat, cam_embeds):
        """
        :param x: triplane features [batch, triplane_features_res, hidden_dim]
        :param img_feat: Image features [batch, seq_len, image_feat_dim]
        :param cam_embeds: Camera Embeddings [batch, camera_dims]
        :return: triplane_hidden_state [batch, triplane_features_res, hidden_dim]
        """
        x = x + self.cross_attn(self.modulated_norm_1(x, cam_embeds), img_feat, img_feat)[0]
        y = self.modulated_norm_2(x, cam_embeds)
        x = x + self.self_attn(y, y, y)[0]
        x = x + self.mlp(self.modulated_norm_3(x, cam_embeds))
        return x


class Decoder(nn.Module):
    def __init__(self, hidden_dim, image_feat_dim, camera_embed_dim, triplane_features_res, triplane_res, triplane_dim,
                 num_layers, num_heads, eps = 1e-6):
        super(Decoder, self).__init__()

        self.triplane_features_res = triplane_features_res  # before up-sampling
        self.triplane_res = triplane_res  # after up-sampling
        self.triplane_dim = triplane_dim

        # initializing triplane features
        self.pos_embed = nn.Parameter(
            torch.randn(1, 3 * triplane_features_res ** 2, hidden_dim) * (1. / hidden_dim) ** 0.5)

        self.layers = nn.ModuleList([
            Block(
                hidden_dim=hidden_dim, cond_dim=image_feat_dim, mod_dim=camera_embed_dim, num_heads=num_heads, eps=eps)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.upsample = nn.ConvTranspose2d(hidden_dim, triplane_dim, kernel_size=2, stride=2, padding=0)

    def forward(self, image_feats, camera_embeddings):
        """
        :param image_feats: [batch, seq_len, dim]
        :param camera_embeddings: [batch, camera_dim]
        :return:
        """ 
        assert image_feats.shape[0] == camera_embeddings.shape[0], \
            f"Mismatched batch size: {image_feats.shape[0]} vs {camera_embeddings.shape[0]}"

        N = image_feats.shape[0]
        H = W = self.triplane_features_res
        L = 3 * H * W

        x = self.pos_embed.repeat(N, 1, 1)  # [N, L, D]
        for layer in self.layers:
            x = layer(x, image_feats, camera_embeddings)
        x = self.norm(x)

        # separate each plane and apply upsampling transposed conv
        x = x.view(N, 3, H, W, -1)
        x = torch.einsum('nihwd->indhw', x)  # [3, N, D, H, W]
        x = x.contiguous().view(3 * N, -1, H, W)  # [3*N, D, H, W]
        x = self.upsample(x)  # [3*N, D', H', W']
        x = x.view(3, N, *x.shape[-3:])  # [3, N, D', H', W']
        x = torch.einsum('indhw->nidhw', x)  # [N, 3, D', H', W']
        x = x.contiguous()

        assert self.triplane_res == x.shape[-2], \
            f"Unexpected triplane resolution after upsampling: {x.shape[-2]} vs {self.triplane_res}"
        assert self.triplane_dim == x.shape[-3], \
            f"Unexpected triplane dimensions after upsampling: {x.shape[-3]} vs {self.triplane_dim}"

        return x