from glob import glob
import os
import torch
import torch.nn as nn
import piqa
import clip
import numpy as np
import wandb
import tqdm
from PIL import Image
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from torch.utils.data import DataLoader
import bitsandbytes as bnb

# Distributed Training imports
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group

from config import Config
from models import MIRANet
from combined_loss import ReconstructionLoss
from dataset import TrainObjaverseDataset, ValidObjaverseDataset

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(config, local_rank, rank):
    if local_rank == 0:
        print("Configuration")
        print(config)

    ckpt = None
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    beta2 = 0.95
    fx = fy = config.focal_length
    px = py = config.principal_point
    k = config.supervision_k

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(config.source_size, scale=(0.5, 1.0)),
        transforms.ToTensor()
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize((config.source_size, config.source_size)),
        transforms.ToTensor()
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TrainObjaverseDataset(config.data_dir, train_transforms)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                                  sampler=DistributedSampler(train_ds, shuffle=True), pin_memory=True)

    val_ds = ValidObjaverseDataset(config.data_dir, valid_transforms)
    # valid_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    model = MIRANet(config.camera_embed_dim, config.decoder_hidden_dim, config.num_layers, config.num_heads,
                    config.triplane_feat_res, config.triplane_res, config.triplane_dim,
                    config.rendering_samples_per_ray,
                    config.camera_matrix_dim).to(device)

    if config.model_save_path and os.path.exists(config.model_save_path) and len(
            glob(f"{config.model_save_path}/*.pt")) > 0:
        if config.model_preloading_strategy == "latest":
            print("Preloading model from latest checkpoint")
            ckpt_path = sorted(glob(f"{config.model_save_path}/*.pt"))[-1]
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt['model_state_dict'])
            config.start_epoch = ckpt['epoch']

    # DDP model
    model = DistributedDataParallel(model, device_ids=[local_rank])

    # optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, beta2))
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = bnb.optim.Adam8bit(optimizer_grouped_parameters, lr=learning_rate, betas=(0.9, beta2))
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                module, 'weight', {'optim_bits': 32}
            )

    if ckpt is not None and isinstance(ckpt, dict):
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader) * config.num_epochs)
    loss_fn = ReconstructionLoss(lambda_value=2.0).to(device)
    scaler = GradScaler()

    # ssim_fn = piqa.SSIM().to(device)
    # psnr_fn = piqa.PSNR()
    # val_model, preprocess = clip.load('ViT-B/32', device=device)

    if rank == 0:
        wandb.init(project='Text to 3D reconstruction', entity='rootacess')
        wandb.config = {
            "learning_rate": learning_rate,
            "epochs": config.num_epochs,
            "batch_size": batch_size,
            **config.to_dict()
        }

    # def compute_clip_similarity(predicted_images, target_images, device):
    #     predicted_images = predicted_images * 255
    #     target_images = target_images * 255
    #     predicted_images = [
    #         preprocess(Image.fromarray(img.cpu().permute(1, 2, 0).numpy().astype(np.uint8))).unsqueeze(0).to(device) for
    #         img
    #         in predicted_images]
    #     target_images = [
    #         preprocess(Image.fromarray(img.cpu().permute(1, 2, 0).numpy().astype(np.uint8))).unsqueeze(0).to(device) for
    #         img
    #         in target_images]
    #
    #     with torch.no_grad():
    #         predicted_features = torch.cat([val_model.encode_image(img) for img in predicted_images])
    #         target_features = torch.cat([val_model.encode_image(img) for img in target_images])
    #
    #     predicted_features_norm = predicted_features / predicted_features.norm(dim=1, keepdim=True)
    #     target_features_norm = target_features / target_features.norm(dim=1, keepdim=True)
    #     similarity = (predicted_features_norm * target_features_norm).sum(dim=1)
    #     return similarity.mean().item()

    # def validate(model, data_loader):
    #     model.eval()
    #     total_ssim, total_psnr, total_clip_similarity = 0, 0, 0
    #     with torch.inference_mode():
    #         data_bar = tqdm.tqdm(data_loader, total=len(data_loader), leave=False, position=1)
    #         for batch in data_bar:
    #             images, cameras = batch
    #             src_cam = torch.cat([cameras, torch.tensor([[fx, fy, px, py]]).repeat(cameras.shape[0], 1)], dim=1)
    #             render_cams = torch.cat([cameras, torch.tensor([fx, 0, px, 0, fy, py, 0, 0, 1]).
    #                                     repeat(cameras.shape[0], 1)], dim=1).unsqueeze(1)
    #
    #             images = images.to(device)
    #             src_cam = src_cam.to(device)
    #             render_cams = render_cams.to(device)
    #             preds = model(images, src_cam, render_cams, config.render_size)['images_rgb'].squeeze(1)
    #
    #             total_psnr += psnr_fn(torch.clamp(preds, min=0., max=1.), images)
    #             total_ssim += ssim_fn(torch.clamp(preds, min=0., max=1.), images)
    #             total_clip_similarity += compute_clip_similarity(preds, images, device=device)
    #
    #     avg_ssim = total_ssim / len(data_loader)
    #     avg_psnr = total_psnr / len(data_loader)
    #     avg_clip_similarity = total_clip_similarity / len(data_loader)
    #     return avg_ssim.item(), avg_psnr.item(), avg_clip_similarity

    def train(model, data_loader, optimizer, scaler):
        model.train()
        total_loss = 0
        data_bar = tqdm.tqdm(data_loader, total=len(data_loader), leave=False, position=1)
        for batch in data_bar:
            images, cameras = batch
            input_image = images[:, 0]
            src_cam = cameras[:, 0]
            src_cam = torch.cat([src_cam, torch.tensor([[fx, fy, px, py]]).repeat(cameras.shape[0], 1)], dim=1)
            intrinsics = torch.tensor([[fx, 0, px, 0, fy, py, 0, 0, 1]]).repeat(4, 1).unsqueeze(0)
            render_cams = torch.cat([cameras, intrinsics.repeat(cameras.shape[0], 1, 1)], dim=2)
            input_image = input_image.to(device)
            src_cam = src_cam.to(device)
            render_cams = render_cams.to(device)

            with autocast():
                preds = model(input_image, src_cam, render_cams, config.render_size)['images_rgb']
                preds = preds.view(preds.shape[0] * k, *preds.shape[2:])
                loss = loss_fn(preds, images.view(images.shape[0] * k, *images.shape[2:]).to(device))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            data_bar.set_description(f"train_batch_loss: {loss.item():.4f}")
        avg_loss = total_loss / len(data_loader)
        return avg_loss

    print("Training Started")

    start_epoch = getattr(config, "start_epoch", 0)
    epoch_progress = tqdm.tqdm(range(start_epoch, config.num_epochs), total=config.num_epochs, leave=True, position=0)
    for epoch in epoch_progress:
        torch.cuda.empty_cache()

        train_loss = train(model.module, train_dataloader, optimizer, scaler)

        if rank == 0:
            # not validating on distributed setup as it can make rest machine stay idle
            # ssim, psnr, clip_similarity = validate(model.module, valid_dataloader)
            # wandb.log({"ssim": ssim, "psnr": psnr, "clip_similarity": clip_similarity})
            wandb.log({"train_loss": train_loss})

        if rank == 0 and (epoch + 1) % config.save_every_epoch == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'wandb_run_id': wandb.run.id
            }, f"{config.model_save_path}/checkpoint_{epoch}.pt")

        lr_scheduler.step()
        epoch_progress.set_description(f"Epoch {epoch + 1}/{config.num_epochs} - train_loss: {train_loss:.4f}")

    return model, optimizer


if __name__ == "__main__":
    # setting up distributed training
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    rank = int(os.environ.get('RANK', -1))

    assert local_rank != -1, "LOCAL_RANK environment variable not set"
    assert rank != -1, "RANK environment variable not set"

    init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    config = Config.from_json(file_path="train_config.json")
    os.makedirs(config.model_save_path, exist_ok=True)
    assert config.num_epochs % config.save_every_epoch == 0, "Total epochs must be divisible by save_every_epoch"

    final_model, optimizer = main(config, local_rank, rank)
    os.makedirs("final_model_checkpoints", exist_ok=True)
    if rank == 0:
        torch.save({
            'epoch': config.num_epochs,
            'model_state_dict': final_model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'wandb_run_id': wandb.run.id
        }, f"final_model_checkpoints/checkpoint_FINAL_{config.num_epochs}.pt")

        wandb.finish()

    destroy_process_group()
