from __future__ import annotations

from typing import Dict, Optional, Tuple, Union, TypedDict

import math
import torch
from torch import nn
from einops import rearrange


# ---------------------------------------------------------------------
# Minimal helpers (avoid hard dependency on timm)
# ---------------------------------------------------------------------
def to_2tuple(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    return (x, x) if isinstance(x, int) else x

def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    if hasattr(nn.init, "trunc_normal_"):
        return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

    with torch.no_grad():

        def norm_cdf(x: torch.Tensor) -> torch.Tensor:
            return (1.0 + torch.erf(x / math.sqrt(2.0))) / 2.0

        low = norm_cdf(torch.tensor((a - mean) / std, device=tensor.device, dtype=tensor.dtype))
        high = norm_cdf(torch.tensor((b - mean) / std, device=tensor.device, dtype=tensor.dtype))
        tensor.uniform_(low, high)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Stochastic depth. Drops entire residual paths."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


# ---------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------
class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"window_partition expects 4D input (B,H,W,C), got {tuple(x.shape)}")
    B, H, W, C = x.shape
    if H % window_size != 0 or W % window_size != 0:
        raise ValueError(f"H and W must be divisible by window_size. Got H={H}, W={W}, window_size={window_size}")
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    if H % window_size != 0 or W % window_size != 0:
        raise ValueError(f"H and W must be divisible by window_size. Got H={H}, W={W}, window_size={window_size}")

    nW = (H // window_size) * (W // window_size)
    if windows.shape[0] % nW != 0:
        raise ValueError(f"windows.shape[0]={windows.shape[0]} must be divisible by nW={nW}")
    B = windows.shape[0] // nW

    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads

        head_dim = dim // num_heads
        if head_dim * num_heads != dim:
            raise ValueError(f"dim must be divisible by num_heads, got dim={dim}, num_heads={num_heads}")
        self.scale = qk_scale or head_dim**-0.5

        # Relative position bias table: (2*Wh-1)*(2*Ww-1), nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # Pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, N, N
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N, N, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # N, N
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (nH, N, N)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 6,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        if not (0 <= self.shift_size < self.window_size):
            raise ValueError("shift_size must satisfy 0 <= shift_size < window_size")

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1

            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )

            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # (nW, ws, ws, 1)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        if L != H * W:
            raise ValueError(f"Input feature has wrong size: L={L}, expected H*W={H*W}")

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)  # (nW*B, ws, ws, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # (nW*B, ws*ws, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # (nW*B, ws*ws, C)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # (B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchExpand(nn.Module):
    def __init__(
        self,
        input_resolution: Tuple[int, int],
        dim: int,
        dim_scale: int = 2,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, dim_scale * dim, bias=False)
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H*W, C)
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        if L != H * W:
            raise ValueError(f"Input feature has wrong size: L={L}, expected H*W={H*W}")

        x = x.view(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c) -> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        return x


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Union[float, list[float]] = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class VideoGenerator(nn.Module):
    def __init__(
        self,
        dim_z: int = 96,
        video_length: int = 50,
        n_channels: int = 1,
        latent_dim: int = 2304,
        bottom_width: int = 6,
        embed_dim: int = 64,
        depths: Tuple[int, int, int, int] = (2, 2, 6, 2),
        num_heads: Tuple[int, int, int, int] = (4, 4, 4, 2),
        window_size: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()

        self.dim_z = dim_z
        self.video_length = video_length
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        self.bottom_width = bottom_width
        self.embed_dim = embed_dim

        self.recurrent = nn.GRUCell(self.dim_z, self.dim_z)
        self.noise_aug = nn.Linear(self.dim_z, self.latent_dim)

        self.blocks_1 = BasicLayer(
            dim=self.embed_dim,
            input_resolution=(self.bottom_width, self.bottom_width),
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=None,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0.0,
            norm_layer=norm_layer,
        )
        self.blocks_2 = BasicLayer(
            dim=self.embed_dim // 2,
            input_resolution=(self.bottom_width * 2, self.bottom_width * 2),
            depth=depths[1],
            num_heads=num_heads[1],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=None,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0.0,
            norm_layer=norm_layer,
        )
        self.blocks_3 = BasicLayer(
            dim=self.embed_dim // 4,
            input_resolution=(self.bottom_width * 4, self.bottom_width * 4),
            depth=depths[2],
            num_heads=num_heads[2],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=None,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0.0,
            norm_layer=norm_layer,
        )
        self.blocks_4 = BasicLayer(
            dim=self.embed_dim // 8,
            input_resolution=(self.bottom_width * 8, self.bottom_width * 8),
            depth=depths[3],
            num_heads=num_heads[3],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=None,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0.0,
            norm_layer=norm_layer,
        )

        self.upsample_1 = PatchExpand(
            input_resolution=(self.bottom_width, self.bottom_width),
            dim=embed_dim,
            dim_scale=2,
            norm_layer=nn.LayerNorm,
        )
        self.upsample_2 = PatchExpand(
            input_resolution=(self.bottom_width * 2, self.bottom_width * 2),
            dim=embed_dim // 2,
            dim_scale=2,
            norm_layer=nn.LayerNorm,
        )
        self.upsample_3 = PatchExpand(
            input_resolution=(self.bottom_width * 4, self.bottom_width * 4),
            dim=embed_dim // 4,
            dim_scale=2,
            norm_layer=nn.LayerNorm,
        )

        self.make_image = nn.Conv2d(
            in_channels=self.embed_dim // 8,
            out_channels=self.n_channels,
            kernel_size=8,
            stride=1,
            padding=0,
        )

    def get_gru_initial_state(
        self, batch_size: int, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        return torch.randn(batch_size, self.dim_z, dtype=dtype, device=device)

    def _encode_input(self, z: torch.Tensor) -> torch.Tensor:
        latent_size = z.size(-1)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-12) * (latent_size**0.5)
        return z

    def _step_fused(self, e_t: torch.Tensor, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_t = self.recurrent(e_t, h_t)
        fused = h_t
        return fused, h_t

    def _project_to_latent(self, fused: torch.Tensor) -> torch.Tensor:
        return self.noise_aug(fused)

    def _decode_fused(self, fused_flat: torch.Tensor) -> torch.Tensor:
        N = fused_flat.shape[0]
        x = fused_flat.view(N, self.bottom_width**2, self.embed_dim)

        x = self.blocks_1(x)
        x = self.upsample_1(x)
        x = self.blocks_2(x)
        x = self.upsample_2(x)
        x = self.blocks_3(x)
        x = self.upsample_3(x)
        x = self.blocks_4(x)

        H = self.bottom_width * 8
        W = self.bottom_width * 8
        C1 = x.shape[-1]
        x = x.permute(0, 2, 1).contiguous().view(N, C1, H, W)
        img = self.make_image(x)
        return img

    class StreamState(TypedDict):
        t: int
        h_t: torch.Tensor

    def init_stream_state(
            self, batch_size: int, device: torch.device, h0: Optional[torch.Tensor] = None, t0: int = 0
    ) -> StreamState:
        if h0 is None:
            h_t = self.get_gru_initial_state(batch_size, device=device)
        else:
            h_t = h0.to(device)
        return {"t": int(t0), "h_t": h_t}

    def forward(
        self,
        z: torch.Tensor,
        streaming: bool = False,
        state: Optional[Dict[str, torch.Tensor]] = None,
        return_state: bool = False,
    ):
        if streaming:
            if z.dim() == 1:
                z = z.unsqueeze(0)
            if z.dim() != 2:
                raise ValueError("Streaming mode expects z with shape [B, dim_z] or [dim_z].")

            B = z.shape[0]
            device = z.device
            if state is None:
                state = self.init_stream_state(B, device=device, t0=0)

            t = int(state.get("t", 0))
            h_t = state["h_t"].to(device)

            e_t = self._encode_input(z)
            fused, h_t = self._step_fused(e_t, h_t)
            latent = self._project_to_latent(fused)
            img_frame = self._decode_fused(latent)

            new_state = {"t": int(t + 1), "h_t": h_t}
            return (img_frame, new_state) if return_state else img_frame

        if z.dim() != 3:
            raise ValueError("Non-streaming mode expects z with shape [B, T, dim_z].")

        B, T_in, D = z.shape
        if D != self.dim_z:
            raise ValueError(f"Expected dim_z={self.dim_z}, got {D}.")

        if self.video_length is None:
            video_len = T_in
        else:
            if T_in < self.video_length:
                raise ValueError(f"Input sequence length T={T_in} is smaller than video_length={self.video_length}.")
            video_len = self.video_length

        z = self._encode_input(z)
        h_t = self.get_gru_initial_state(B, device=z.device, dtype=z.dtype)

        fused_list = []
        for t in range(video_len):
            e_t = z[:, t, :].view(-1, self.dim_z)
            fused, h_t = self._step_fused(e_t, h_t)
            fused_list.append(fused)

        fused_all = torch.stack(fused_list, dim=1)  # [B, T, dim_z]
        latent_all = self._project_to_latent(fused_all)  # [B, T, latent_dim]

        img_flat = self._decode_fused(latent_all.reshape(B * video_len, self.latent_dim))
        C = img_flat.shape[1]
        H, W = img_flat.shape[-2], img_flat.shape[-1]
        video_seq = img_flat.view(B, video_len, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        return img_flat, video_seq
