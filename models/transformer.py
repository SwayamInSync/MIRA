import torch
import torch.nn as nn

from models.encoder.dino import DinoEncoder
from models.decoder.TriplaneDecoder import Decoder
from .volume_rendering.renderer import TriplaneSynthesizer


class CameraEmbeddings(nn.Module):
    def __init__(self, cam_matrix_dim, embed_dim): # (20, 1024)
        super(CameraEmbeddings, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cam_matrix_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class Transformer(nn.Module):
    def __init__(self, camera_embed_dim, hidden_dim, num_layers, num_heads,
                 triplane_feat_res, triplane_res, triplane_dim, cam_matrix_dim, encoder_freeze=True):
        super(Transformer, self).__init__()

        self.encoder_features_dim = 768
        self.camera_embed_dim = camera_embed_dim

        self.encoder = DinoEncoder(encoder_freeze)
        self.camera_embeddings = CameraEmbeddings(cam_matrix_dim=cam_matrix_dim, embed_dim=camera_embed_dim)
        self.decoder = Decoder(hidden_dim, self.encoder_features_dim, camera_embed_dim, triplane_feat_res, triplane_res,
                               triplane_dim, num_layers, num_heads)

    def forward(self, image, camera):
        """
        :param image: [N, C, H, W]
        :param camera: [N, cam_matrix_dim]
        :return:
        """
        assert image.shape[0] == camera.shape[0], "Batch size mismatch for image and camera"
        image_feats = self.encoder(image)
        camera_embeddings = self.camera_embeddings(camera)
        planes = self.decoder(image_feats, camera_embeddings)
        assert planes.shape[1] == 3, "3 channels are needed for planes"
        return planes


class MIRANet(nn.Module):
    def __init__(self, camera_embed_dim, hidden_dim, num_layers, num_heads,
                 triplane_feat_res, triplane_res, triplane_dim, rendering_samples_per_ray, camera_matrix_dim):
        super(MIRANet, self).__init__()
        self.transformer = Transformer(camera_embed_dim, hidden_dim, num_layers, num_heads, triplane_feat_res,
                                       triplane_res, triplane_dim, camera_matrix_dim)
        self.renderer = TriplaneSynthesizer(triplane_dim, rendering_samples_per_ray)

    def forward(self, image, source_camera, render_cameras, render_size):
        planes = self.transformer(image, source_camera)
        # print("rendering")
        render_results = self.renderer(planes, render_cameras, render_size)
        return {
            'planes': planes,
            **render_results,
        }


if __name__ == "__main__":
    from config import Config

    model_config = Config.from_json("../final_train_config.json")

    img = torch.randn(1, 3, model_config.source_size, model_config.source_size)
    src_cam = torch.randn(1, model_config.camera_matrix_dim)
    render_cams = torch.randn(1, 2, 25)
    render_size = model_config.render_size

    net = MIRANet(model_config.camera_embed_dim, model_config.decoder_hidden_dim,
                  model_config.num_layers, model_config.num_heads, model_config.triplane_feat_res,
                  model_config.triplane_res, model_config.triplane_dim, model_config.rendering_samples_per_ray,
                  model_config.camera_matrix_dim)
    op = net(img, src_cam, render_cams, render_size)
    for k, v in op.items():
        print(k, v.shape, v.max(), v.min())
    print("success")
    print([i for i in net.modules()])

# Output:
# planes torch.Size([1, 3, 40, 64, 64])
# images_rgb torch.Size([1, 2, 3, 192, 192])
# images_depth torch.Size([1, 2, 1, 192, 192])
# images_weight torch.Size([1, 2, 1, 192, 192])
