import math
import torch
import argparse
import numpy as np
# from PIL import Image
import cv2 as cv
import mcubes
import trimesh
import os
import imageio

from models import MIRANet
from config import Config


def images_to_video(images, output_path, fps):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames = []
    for i in range(images.shape[0]):
        frame = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        assert frame.shape[0] == images.shape[2] and frame.shape[1] == images.shape[3], \
            f"Frame shape mismatch: {frame.shape} vs {images.shape}"
        assert frame.min() >= 0 and frame.max() <= 255, \
            f"Frame value out of range: {frame.min()} ~ {frame.max()}"
        frames.append(frame)
    imageio.mimwrite(output_path, np.stack(frames), fps=fps, codec='mpeg4', quality=10)
    print(f"Saved video to {output_path}")


class MIRAInference:
    def __init__(self, args: argparse.Namespace):
        # assert args.checkpoint_path is not None and config_path is not None, "checkpoint_path and config_path is
        # required"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_config = Config.from_json(args.config_path)
        self.args = args

        self.model = MIRANet(self.model_config.camera_embed_dim, self.model_config.decoder_hidden_dim,
                             self.model_config.num_layers, self.model_config.num_heads,
                             self.model_config.triplane_feat_res,
                             self.model_config.triplane_res, self.model_config.triplane_dim,
                             self.model_config.rendering_samples_per_ray,
                             self.model_config.camera_matrix_dim).to(self.device).eval()

        if args.checkpoint_path is not None:
            print(f"Loading the model weights from {self.args.checkpoint_path}")
            self.model.load_state_dict(torch.load(self.args.checkpoint_path)['model_state_dict'])

    def text_mode(self, prompt):
        image = None  # todo: Use SDXL for image generation
        self.image_mode(image)

    def image_mode(self, image):
        if isinstance(image, str):
            image = torch.tensor(cv.cvtColor(cv.imread(image), cv.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0) / 255.0
        image = torch.nn.functional.interpolate(image,
                                                size=(self.model_config.source_size, self.model_config.source_size),
                                                mode='bicubic', align_corners=True)
        image = torch.clamp(image, 0, 1)
        results = self.reconstruct(image, self.model_config.render_size, self.args.mesh_size, self.args.export_video,
                                   self.args.export_mesh)

        output_id = "Output"
        os.makedirs(self.args.output_path, exist_ok=True)

        if 'frames' in results:
            renderings = results['frames']
            frames = renderings.get('images_rgb', None)
            if frames is not None:
                images_to_video(frames[0], os.path.join(self.args.output_path, f'{output_id}.mov'), fps=40)
            print(f"Video saved to {self.args.output_path}")

        if 'mesh' in results:
            mesh = results['mesh']
            mesh.export(os.path.join(self.args.output_path, f'{output_id}.ply'), 'ply')
            print(f"Mesh dumped to {self.args.output_path}")

    def reconstruct(self, image, render_size, mesh_size, export_video, export_mesh):
        mesh_thres = 3.0
        chunk_size = 2
        batch_size = 1

        src_cam = self._default_source_camera(batch_size).to(self.device)
        render_cams = self._default_render_cameras(batch_size).to(self.device)

        with torch.inference_mode():
            planes = self.model.transformer(image, src_cam)
            results = {}

            if export_video:
                frames = []
                for i in range(0, render_cams.shape[1], chunk_size):
                    frames.append(
                        self.model.renderer(
                            planes,
                            render_cams[:, i:i + chunk_size],
                            render_size,
                        )
                    )

                # merge frames
                frames = {
                    k: torch.cat([r[k] for r in frames], dim=1)
                    for k in frames[0].keys()
                }
                # update results
                results.update({
                    'frames': frames,
                })

            if export_mesh:
                grid_out = self.model.renderer.forward_grid(
                    planes=planes,
                    grid_size=mesh_size,
                )

                vtx, faces = mcubes.marching_cubes(grid_out['sigma'].squeeze(0).squeeze(-1).cpu().numpy(), mesh_thres)
                vtx = vtx / (mesh_size - 1) * 2 - 1

                vtx_tensor = torch.tensor(vtx, dtype=torch.float32, device=self.device).unsqueeze(0)
                vtx_colors = self.model.renderer.forward_points(planes, vtx_tensor)['rgb'].squeeze(
                    0).cpu().numpy()  # (0, 1)
                vtx_colors = (vtx_colors * 255).astype(np.uint8)

                mesh = trimesh.Trimesh(vertices=vtx, faces=faces, vertex_colors=vtx_colors)

                results.update({
                    'mesh': mesh,
                })

            return results

    def _default_source_camera(self, batch_size=1) -> torch.Tensor:
        dist_to_center = 2
        camera_extrinsics = torch.tensor([[
            [1, 0, 0, 0],
            [0, 0, -1, -dist_to_center],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]], dtype=torch.float32)
        fx = fy = torch.tensor([self.model_config.focal_length], dtype=torch.float32)
        cx = cy = torch.tensor([self.model_config.principal_point], dtype=torch.float32)
        source_camera = torch.cat([
            camera_extrinsics.reshape(-1, 16),
            fx.unsqueeze(-1), fy.unsqueeze(-1), cx.unsqueeze(-1), cy.unsqueeze(-1),
        ], dim=-1)
        return source_camera.repeat(batch_size, 1)

    def _default_render_cameras(self, batch_size=1, num_cameras=160, radius=2.0, height=0.8) -> torch.Tensor:
        # generating surrounding views
        camera_positions = []
        projected_radius = math.sqrt(radius ** 2 - height ** 2)
        for i in range(num_cameras):
            theta = ((2 * math.pi * i) / num_cameras) - (math.pi / 2)
            x = projected_radius * math.cos(theta)
            y = projected_radius * math.sin(theta)
            z = height
            camera_positions.append([x, y, z])
        camera_positions = torch.tensor(camera_positions, dtype=torch.float32)

        # center looking at camera
        look_at = torch.tensor([0, 0, 0], dtype=torch.float32)
        up_world = torch.tensor([0, 0, 1], dtype=torch.float32)
        look_at = look_at.unsqueeze(0).repeat(camera_positions.shape[0], 1)
        up_world = up_world.unsqueeze(0).repeat(camera_positions.shape[0], 1)

        z_axis = camera_positions - look_at
        z_axis = z_axis / z_axis.norm(dim=-1, keepdim=True)
        x_axis = torch.cross(up_world, z_axis)
        x_axis = x_axis / x_axis.norm(dim=-1, keepdim=True)
        y_axis = torch.cross(z_axis, x_axis)
        y_axis = y_axis / y_axis.norm(dim=-1, keepdim=True)
        extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_positions], dim=-1)

        fx = fy = torch.tensor([self.model_config.focal_length], dtype=torch.float32)
        cx = cy = torch.tensor([self.model_config.principal_point], dtype=torch.float32)

        extrinsics = torch.cat([extrinsics,
                                torch.tensor([[[0, 0, 0, 1]]], dtype=torch.float32).
                               repeat(extrinsics.shape[0], 1, 1)], dim=1)
        intrinsics = torch.tensor([[fx, torch.zeros_like(fx), cx, torch.zeros_like(fy), fy, cy, 0, 0, 1]],
                                  dtype=torch.float32, device=extrinsics.device).repeat(extrinsics.shape[0], 1)
        render_cameras = torch.cat([extrinsics.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=-1)
        return render_cameras.unsqueeze(0).repeat(batch_size, 1, 1)

    def __call__(self, mode, input_):
        if mode == 'text':
            self.text_mode(input_)
        elif mode == 'image':
            self.image_mode(input_)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--config_path', type=str, default="train_config.json")
    parser.add_argument('--mode', type=str, default='image', help="Support two modes 'text' and 'image'")
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output_path', type=str, default='output')
    parser.add_argument('--mesh_size', type=int, default=384)
    parser.add_argument('--export_video', type=bool, default=True)
    parser.add_argument('--export_mesh', type=bool, default=True)
    args = parser.parse_args()
    print(vars(args))
    infer = MIRAInference(args)
    infer(args.mode, args.input)
