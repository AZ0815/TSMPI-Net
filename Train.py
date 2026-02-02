from __future__ import annotations

import os
import random
from typing import List

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from Dataset import train_dataset
from Generator import VideoGenerator
from Discriminator_Frame_level import Discriminator_Frame
from Discriminator_Sequence_level import Discriminator_Sequence

try:
    from timm.models.layers import trunc_normal_ as _timm_trunc_normal_
except Exception:
    _timm_trunc_normal_ = None


# -------------------------
# User Config
# -------------------------
SEED = 999
NUM_EPOCHS = 100
BATCH_SIZE = 8
NUM_WORKERS = 4

VIDEO_LEN = 50
LAMBDA_DIST_PENALTY = 20

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "./weights")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    """Truncated normal init with a safe fallback."""
    if _timm_trunc_normal_ is not None:
        return _timm_trunc_normal_(tensor, std=std)
    if hasattr(nn.init, "trunc_normal_"):
        return nn.init.trunc_normal_(tensor, std=std)
    with torch.no_grad():
        tensor.normal_(0.0, std)
        tensor.clamp_(-2.0 * std, 2.0 * std)
    return tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_train_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """Build DataLoader on the full dataset (no subset sampling)."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        _trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class generatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fake: torch.Tensor) -> torch.Tensor:
        return -torch.mean(fake)


class discriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        relu = nn.ReLU(inplace=True)
        real_loss = torch.mean(relu(1.0 - real))
        fake_loss = torch.mean(relu(1.0 + fake))
        return (real_loss + fake_loss) * 0.5


def save_checkpoint(
    path: str,
    models: List[nn.Module],
    optimizers: List[optim.Optimizer],
    epoch: int,
    last_batch_idx: int,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "last_batch_idx": last_batch_idx,
        "model": [models[0].state_dict(), models[1].state_dict(), models[2].state_dict()],
        "optimizer": [
            optimizers[0].state_dict(),
            optimizers[1].state_dict(),
            optimizers[2].state_dict(),
        ],
    }
    torch.save(state, path)


def train(
    dataloader: DataLoader,
    gen: nn.Module,
    disc_3d: nn.Module,
    disc_2d: nn.Module,
    opt_gen: optim.Optimizer,
    opt_disc1: optim.Optimizer,
    opt_disc2: optim.Optimizer,
    criterion_g: nn.Module,
    criterion_d: nn.Module,
    num_epochs: int = NUM_EPOCHS,
    video_len: int = VIDEO_LEN,
    lambda_dist_penalty: float = LAMBDA_DIST_PENALTY,
    checkpoint_dir: str = CHECKPOINT_DIR,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        gen.train()
        disc_2d.train()
        disc_3d.train()

        last_batch_idx = len(dataloader) - 1

        for batch_index, (data_1d, data_2d) in enumerate(dataloader):
            signal_seq = data_1d.to(DEVICE)
            real_img_video = data_2d.to(DEVICE)

            if real_img_video.shape[2] != video_len:
                raise ValueError(
                    f"Expected video length {video_len}, but got {real_img_video.shape[2]}"
                )

            B = real_img_video.shape[0]
            C = real_img_video.shape[1]
            H = real_img_video.shape[3]
            W = real_img_video.shape[4]

            fake_img_seq, fake_img_video = gen(signal_seq)
            real_img_seq = (
                real_img_video.permute(0, 2, 1, 3, 4)
                .contiguous()
                .view(B * video_len, C, H, W)
                .to(DEVICE)
            )

            disc_2d.zero_grad()
            disc_real = disc_2d(real_img_seq).flatten()
            disc_fake = disc_2d(fake_img_seq.detach().clone()).flatten()
            loss_disc = criterion_d(disc_fake, disc_real)
            loss_disc.backward()
            opt_disc2.step()

            disc_3d.zero_grad()
            disc_real_seq = disc_3d(real_img_video).flatten()
            disc_fake_seq = disc_3d(fake_img_video.detach().clone()).flatten()
            loss_disc_seq = criterion_d(disc_fake_seq, disc_real_seq)
            loss_disc_seq.backward()
            opt_disc1.step()

            gen.zero_grad()
            output_seq = disc_3d(fake_img_video).flatten()
            loss_gen_seq = criterion_g(output_seq)

            output = disc_2d(fake_img_seq).flatten()
            loss_gen = criterion_g(output)

            loss_g_dist = torch.mean(torch.abs(real_img_seq - fake_img_seq)) * lambda_dist_penalty
            loss_g_main = (loss_gen_seq + loss_gen) * 0.5
            loss_gen_sum = loss_g_main + loss_g_dist

            loss_gen_sum.backward()
            opt_gen.step()

        ckpt_path = os.path.join(checkpoint_dir, f"{epoch}-checkpoint-{last_batch_idx}.pth")
        save_checkpoint(
            ckpt_path,
            models=[gen, disc_3d, disc_2d],
            optimizers=[opt_gen, opt_disc1, opt_disc2],
            epoch=epoch,
            last_batch_idx=last_batch_idx,
        )


def main() -> None:
    set_seed(SEED)

    train_loader = build_train_dataloader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    gen = VideoGenerator().to(DEVICE)
    disc_2d = Discriminator_Frame().to(DEVICE)
    disc_3d = Discriminator_Sequence().to(DEVICE)

    gen.apply(_init_weights)
    disc_2d.apply(_init_weights)
    disc_3d.apply(_init_weights)

    opt_gen = optim.Adam(gen.parameters(), lr=4e-5, betas=(0.5, 0.999))
    opt_disc1 = optim.Adam(disc_3d.parameters(), lr=1e-6, betas=(0.5, 0.999))
    opt_disc2 = optim.Adam(disc_2d.parameters(), lr=1e-6, betas=(0.5, 0.999))

    criterion_g = generatorLoss()
    criterion_d = discriminatorLoss()

    train(
        dataloader=train_loader,
        gen=gen,
        disc_3d=disc_3d,
        disc_2d=disc_2d,
        opt_gen=opt_gen,
        opt_disc1=opt_disc1,
        opt_disc2=opt_disc2,
        criterion_g=criterion_g,
        criterion_d=criterion_d,
    )


if __name__ == "__main__":
    main()
