import os
from glob import glob
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
import bitsandbytes as bnb

from config import Config
from models import Network
from combined_loss import ReconstructionLoss
from dataset import get_loader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

config = Config.from_json(file_path="final_train_config.json")
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

train_dataloader = get_loader("temp_data/render", train_transforms, batch_size)
valid_dataloader = get_loader("temp_data/render", valid_transforms, batch_size, is_valid=True)

model = Network(config.camera_embed_dim, config.decoder_hidden_dim, config.num_layers, config.num_heads,
                config.triplane_feat_res, config.triplane_res, config.triplane_dim, config.rendering_samples_per_ray,
                config.camera_matrix_dim).to(device)

if config.model_save_path and os.path.exists(config.model_save_path):
    if config.model_preloading_strategy == "latest":
        print("Preloading model from latest checkpoint")
        ckpt_path = glob(f"{config.model_save_path}/*.pt")[-1]
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model_state_dict'])
        config.start_epoch = ckpt['epoch']

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
# optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, beta2))

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

ssim_fn = piqa.SSIM().to(device)
psnr_fn = piqa.PSNR()
val_model, preprocess = clip.load('ViT-B/32', device=device)


def compute_clip_similarity(predicted_images, target_images, device):
    # preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
    #                                  transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
    #                                                       (0.26862954, 0.26130258, 0.27577711))])
    predicted_images = predicted_images * 255
    target_images = target_images * 255
    predicted_images = [
        preprocess(Image.fromarray(img.cpu().permute(1, 2, 0).numpy().astype(np.uint8))).unsqueeze(0).to(device) for img
        in predicted_images]
    target_images = [
        preprocess(Image.fromarray(img.cpu().permute(1, 2, 0).numpy().astype(np.uint8))).unsqueeze(0).to(device) for img
        in target_images]

    with torch.no_grad():
        predicted_features = torch.cat([val_model.encode_image(img) for img in predicted_images])
        target_features = torch.cat([val_model.encode_image(img) for img in target_images])

    predicted_features_norm = predicted_features / predicted_features.norm(dim=1, keepdim=True)
    target_features_norm = target_features / target_features.norm(dim=1, keepdim=True)
    similarity = (predicted_features_norm * target_features_norm).sum(dim=1)
    return similarity.mean().item()


def validate(model, data_loader):
    model.eval()
    total_ssim, total_psnr, total_clip_similarity = 0, 0, 0
    with torch.inference_mode():
        data_bar = tqdm.tqdm(data_loader, total=len(data_loader), leave=False, position=1)
        for batch in data_bar:
            images, cameras = batch
            src_cam = torch.cat([cameras, torch.tensor([[fx, fy, px, py]]).repeat(cameras.shape[0], 1)], dim=1)
            render_cams = torch.cat([cameras, torch.tensor([fx, 0, px, 0, fy, py, 0, 0, 1]).
                                    repeat(cameras.shape[0], 1)], dim=1).unsqueeze(1)

            images = images.to(device)
            src_cam = src_cam.to(device)
            render_cams = render_cams.to(device)
            preds = model(images, src_cam, render_cams, config.render_size)['images_rgb'].squeeze(1)

            total_psnr += psnr_fn(torch.clamp(preds, min=0., max=1.), images)
            total_ssim += ssim_fn(torch.clamp(preds, min=0., max=1.), images)
            total_clip_similarity += compute_clip_similarity(preds, images, device=device)

    avg_ssim = total_ssim / len(data_loader)
    avg_psnr = total_psnr / len(data_loader)
    avg_clip_similarity = total_clip_similarity / len(data_loader)
    return avg_ssim.item(), avg_psnr.item(), avg_clip_similarity


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


wandb.init(project='Text to 3D reconstruction', entity='rootacess')
wandb.config = {
    "learning_rate": learning_rate,
    "epochs": config.num_epochs,
    "batch_size": batch_size,
    **config.to_dict()
}

os.makedirs(config.model_save_path, exist_ok=True)
print("Training Started")
start_epoch = getattr(config, "start_epoch", 0)
epoch_progress = tqdm.tqdm(range(start_epoch, config.num_epochs), total=config.num_epochs, leave=True, position=0)
for epoch in epoch_progress:
    torch.cuda.empty_cache()

    train_loss = train(model, train_dataloader, optimizer, scaler)
    ssim, psnr, clip_similarity = validate(model, valid_dataloader)
    wandb.log({"ssim": ssim, "psnr": psnr, "clip_similarity": clip_similarity})
    wandb.log({"train_loss": train_loss})
    lr_scheduler.step()

    if (epoch + 1) % config.save_every_epoch:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'wandb_run_id': wandb.run.id
        }, f"{config.model_save_path}/checkpoint_{epoch}.pt")

    epoch_progress.set_description(f"Epoch {epoch + 1}/{config.num_epochs} - train_loss: {train_loss:.4f}")

os.makedirs("final_model_checkpoints", exist_ok=True)
torch.save({
    'epoch': config.num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'wandb_run_id': wandb.run.id
}, f"final_model_checkpoints/checkpoint_FINAL_{config.num_epochs}.pt")

wandb.finish()
