from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

import torch
from torch import nn, Tensor
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



class SharedSpatialInputAdapter(nn.Module):
    """
    Produces one locality-aware spectrum from an odd-sized spatial patch.

    The same spatial weights are applied independently to every spectral
    band. Spectral bands are never mixed with one another.

    Input:
        x: [batch, height, width, bands]

    Output:
        adapted_spectrum: [batch, bands]
    """

    def __init__(
        self,
        patch_size: int,
        initial_mix: float = 0.02,
    ) -> None:
        super().__init__()

        if patch_size <= 1:
            raise ValueError(
                "SharedSpatialInputAdapter requires patch_size > 1."
            )

        if patch_size % 2 == 0:
            raise ValueError(
                "patch_size must be odd so the patch has a center."
            )

        if not 0.0 < initial_mix < 1.0:
            raise ValueError(
                "initial_mix must be strictly between 0 and 1."
            )

        self.patch_size = patch_size
        self.num_pixels = patch_size * patch_size
        self.center_index = self.num_pixels // 2

        # Store the flattened indices of every pixel except the center.
        neighbor_indices = [
            index
            for index in range(self.num_pixels)
            if index != self.center_index
        ]

        self.register_buffer(
            "neighbor_indices",
            torch.tensor(neighbor_indices, dtype=torch.long),
            persistent=False,
        )

        # One learned scalar per relative neighbor location.
        # Softmax converts these into nonnegative weights summing to one.
        self.neighbor_logits = nn.Parameter(
            torch.zeros(self.num_pixels - 1)
        )

        # alpha = sigmoid(mix_logit).
        #
        # Initialize alpha near zero so training begins close to the
        # original center-pixel HyperSL linear probe.
        initial_mix_tensor = torch.tensor(
            initial_mix,
            dtype=torch.float32,
        )

        initial_logit = torch.log(
            initial_mix_tensor / (1.0 - initial_mix_tensor)
        )

        self.mix_logit = nn.Parameter(initial_logit)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(
                "Expected x with shape [batch, height, width, bands], "
                f"but received {tuple(x.shape)}."
            )

        batch, height, width, bands = x.shape

        if height != self.patch_size or width != self.patch_size:
            raise ValueError(
                f"Expected a {self.patch_size}x{self.patch_size} patch, "
                f"but received {height}x{width}."
            )

        # [B, H*W, C]
        flattened = x.reshape(
            batch,
            self.num_pixels,
            bands,
        )

        # [B, C]
        center = flattened[:, self.center_index, :]

        # [B, H*W-1, C]
        neighbors = flattened.index_select(
            dim=1,
            index=self.neighbor_indices,
        )

        # [H*W-1]
        spatial_weights = torch.softmax(
            self.neighbor_logits,
            dim=0,
        )

        # Fixed across spectral bands:
        # [B, H*W-1, C] × [1, H*W-1, 1] -> [B, C]
        neighborhood_spectrum = (
            neighbors
            * spatial_weights.view(1, -1, 1)
        ).sum(dim=1)

        # Scalar constrained to [0, 1].
        mix = torch.sigmoid(self.mix_logit)

        # Residual interpolation keeps the adapter close to the
        # original center spectrum unless locality proves useful.
        adapted_spectrum = (
            center
            + mix * (neighborhood_spectrum - center)
        )

        return adapted_spectrum

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
        self.patch_size = patch_size

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
        elif self.head_type == "input_adapter_linear":
            self.input_adapter = SharedSpatialInputAdapter(
                patch_size=patch_size,
                initial_mix=0.02,
            )

            self.classifier = nn.Linear(
                self.embedding_dim,
                class_num,
            )
        else:
            raise ValueError(
                "head_type must be one of {'linear', 'cnn', 'local_attention', 'input_adapter_linear'}, "
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
        # if self.encoder_frozen:
        #     with torch.no_grad():
        #         z, _, _, _, _, shape = self.spectral_encoder.encoder_forward(
        #             x, wavelength, 0.0
        #         )
        # else:
        #     z, _, _, _, _, shape = self.spectral_encoder.encoder_forward(
        #         x, wavelength, 0.0
        #     )
        z, _, _, _, _, shape = self.spectral_encoder.encoder_forward(
            x, wavelength, 0.0
        )

        batch, height, width, _ = shape
        # z is [B*H*W, 1, D]. Restore the spatial arrangement.
        feature_map = z.reshape(batch, height, width, self.embedding_dim)
        return feature_map.permute(0, 3, 1, 2).contiguous()

    # def forward(self, x, wavelength):
    def forward(
        self,
        x: torch.Tensor,
        wavelength: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "Expected [B, H, W, C], "
                f"received {tuple(x.shape)}."
            )

        batch, height, width, bands = x.shape

        if height != self.patch_size or width != self.patch_size:
            raise ValueError(
                f"Expected spatial shape "
                f"{self.patch_size}x{self.patch_size}, "
                f"received {height}x{width}."
            )

        if wavelength is not None:
            expected_bands = wavelength.shape[-1]

            if bands != expected_bands:
                raise ValueError(
                    "Input does not appear to be [B, H, W, C]. "
                    f"Last input dimension is {bands}, but wavelength "
                    f"dimension is {expected_bands}. "
                    f"Full input shape: {tuple(x.shape)}."
                )
        
        if self.head_type == "input_adapter_linear":
            # x is [B, H, W, C].
            #
            # The adapter reduces the spatial patch to one spectrum
            # while retaining the original number and ordering of bands.
            # print(x.shape)
            adapted_spectrum = self.input_adapter(x)  # [B, C]
            # print(adapted_spectrum.shape)

            # Reintroduce 1x1 spatial dimensions because encode_patch()
            # expects a spatial-spectral patch.
            adapted_patch = adapted_spectrum[:, None, None, :]
            # print(adapted_patch.shape)
            # exit()
            # Frozen HyperSL produces a [B, D, 1, 1] feature map.
            feature_map = self.encode_patch(
                adapted_patch,
                wavelength,
            )

            features = feature_map.flatten(1)  # [B, D]

            return self.classifier(features)

        # Existing linear/CNN/local-attention cases follow here.
        feature_map = self.encode_patch(x, wavelength)

        if self.head_type == "linear":
            features = self.pool(feature_map).flatten(1)
            return self.classifier(features)
        if self.head_type == "cnn":
            features = self.pool(self.convblock(feature_map)).flatten(1)
            return self.classifier(features)
        return self.local_head(feature_map)
