import torch
from torch import nn
import torch.nn.functional as F

from .model import SpectralSharedEncoder


class SpatialTransformerHead(nn.Module):
    """Model relationships among per-pixel HyperSL embeddings in a local patch."""

    def __init__(
        self,
        embedding_dim: int,
        class_num: int,
        patch_size: int,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        if patch_size % 2 == 0:
            raise ValueError("patch_size must be odd so the patch has a center pixel.")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by spatial num_heads.")

        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.pos_embed = nn.Parameter(
            torch.zeros(1, patch_size * patch_size, embedding_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=int(embedding_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=depth,
            norm=nn.LayerNorm(embedding_dim),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(embedding_dim * 2),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, class_num)

    def _position_embedding(self, height: int, width: int) -> torch.Tensor:
        if height == self.patch_size and width == self.patch_size:
            return self.pos_embed

        # Preserve compatibility with another odd patch size at inference time.
        pos = self.pos_embed.view(
            1, self.patch_size, self.patch_size, self.embedding_dim
        ).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(height, width), mode="bicubic", align_corners=False)
        return pos.permute(0, 2, 3, 1).reshape(1, height * width, self.embedding_dim)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        # feature_map: [B, D, H, W]
        batch, channels, height, width = feature_map.shape
        if channels != self.embedding_dim:
            raise ValueError(
                f"Expected {self.embedding_dim} feature channels, got {channels}."
            )
        if height % 2 == 0 or width % 2 == 0:
            raise ValueError("Spatial dimensions must be odd to identify a center pixel.")

        tokens = feature_map.flatten(2).transpose(1, 2)  # [B, H*W, D]
        tokens = self.encoder(tokens + self._position_embedding(height, width))

        center_index = (height // 2) * width + (width // 2)
        center_feature = tokens[:, center_index]
        neighborhood_feature = tokens.mean(dim=1)
        fused = self.fusion(torch.cat((center_feature, neighborhood_feature), dim=-1))
        return self.classifier(fused)


class ClassificationModel(nn.Module):
    def __init__(
        self,
        class_num,
        model_size,
        embedding_dim=None,
        encoder_depth=None,
        decoder_depth=None,
        num_heads=None,
        use_checkpointing=False,
        linear_probe=False,
        head_type="cnn",
        patch_size=5,
        spatial_depth=2,
        spatial_heads=4,
        spatial_mlp_ratio=4.0,
        spatial_dropout=0.1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.encoder_frozen = False

        # Backward compatibility with the old CLI/API.
        if linear_probe:
            head_type = "linear"
        self.head_type = head_type

        presets = {
            "small": dict(embedding_dim=256, encoder_depth=8, decoder_depth=4, num_heads=8),
            "base": dict(embedding_dim=512, encoder_depth=24, decoder_depth=12, num_heads=16),
            "large": dict(embedding_dim=1024, encoder_depth=32, decoder_depth=16, num_heads=32),
            "huge": dict(embedding_dim=2048, encoder_depth=48, decoder_depth=24, num_heads=32),
        }
        if model_size not in presets:
            raise ValueError(f"Unsupported model_size: {model_size}")

        config = presets[model_size].copy()
        if embedding_dim is not None:
            config["embedding_dim"] = embedding_dim
        if encoder_depth is not None:
            config["encoder_depth"] = encoder_depth
        if decoder_depth is not None:
            config["decoder_depth"] = decoder_depth
        if num_heads is not None:
            config["num_heads"] = num_heads

        self.embedding_dim = config["embedding_dim"]
        self.spectral_encoder = SpectralSharedEncoder(
            embedding_dim=config["embedding_dim"],
            max_band=500,
            encoder_depth=config["encoder_depth"],
            decoder_depth=config["decoder_depth"],
            num_heads=config["num_heads"],
            mlp_ratio=4.0,
            norm_layer=nn.LayerNorm,
            use_checkpointing=use_checkpointing,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        if self.head_type == "linear":
            self.classifier = nn.Linear(self.embedding_dim, class_num)
        elif self.head_type == "cnn":
            self.convblock = nn.Sequential(
                nn.Conv2d(self.embedding_dim, 64, 1, 1),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.Conv2d(64, 512, 3, 2, 1),
                nn.BatchNorm2d(512),
                nn.GELU(),
            )
            self.classifier = nn.Sequential(
                nn.Linear(512, 1024),
                nn.Dropout(0.2),
                nn.GELU(),
                nn.Linear(1024, 1024),
                nn.Dropout(0.4),
                nn.GELU(),
                nn.Linear(1024, class_num),
            )
        elif self.head_type == "local_attention":
            self.local_head = SpatialTransformerHead(
                embedding_dim=self.embedding_dim,
                class_num=class_num,
                patch_size=patch_size,
                depth=spatial_depth,
                num_heads=spatial_heads,
                mlp_ratio=spatial_mlp_ratio,
                dropout=spatial_dropout,
            )
        else:
            raise ValueError(
                "head_type must be one of {'linear', 'cnn', 'local_attention'}, "
                f"got {self.head_type!r}."
            )

    def freeze_encoder(self):
        self.encoder_frozen = True
        for param in self.spectral_encoder.parameters():
            param.requires_grad = False
        self.spectral_encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_frozen:
            self.spectral_encoder.eval()
        return self

    def encode_patch(self, x, wavelength):
        """Return one fixed-dimensional HyperSL embedding per pixel."""
        if self.encoder_frozen:
            with torch.no_grad():
                z, _, _, _, _, shape = self.spectral_encoder.encoder_forward(
                    x, wavelength, 0.0
                )
        else:
            z, _, _, _, _, shape = self.spectral_encoder.encoder_forward(
                x, wavelength, 0.0
            )

        batch, height, width, _ = shape
        # z is [B*H*W, 1, D]. Restore the spatial arrangement.
        feature_map = z.reshape(batch, height, width, self.embedding_dim)
        return feature_map.permute(0, 3, 1, 2).contiguous()

    def forward(self, x, wavelength):
        feature_map = self.encode_patch(x, wavelength)

        if self.head_type == "linear":
            features = self.pool(feature_map).flatten(1)
            return self.classifier(features)
        if self.head_type == "cnn":
            features = self.pool(self.convblock(feature_map)).flatten(1)
            return self.classifier(features)
        return self.local_head(feature_map)
