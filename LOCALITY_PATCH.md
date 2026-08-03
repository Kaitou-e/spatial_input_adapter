# HyperSL locality-aware downstream patch

## What the original code already does

`SpectralSharedEncoder.encoder_forward()` accepts a patch shaped `[B, H, W, C]`, but it reshapes that patch to `B*H*W` independent spectra. HyperSL therefore produces one spectral embedding per pixel without allowing neighboring pixels to communicate inside the encoder.

The original `ClassificationModel` then restores the `[H, W]` grid. Its `cnn` head already uses local information through `Conv2d`, while the existing `linear_probe` path averages embeddings and applies one linear layer. The submission scripts use `--patch-size 1 --linear-probe`, so those runs contain no neighborhood information.

## Added head

This patch adds `head_type=local_attention`:

1. HyperSL encodes every pixel spectrum independently.
2. The embeddings are restored to `[B, D, H, W]`.
3. A small spatial Transformer attends over the `H*W` pixel embeddings.
4. The center-pixel representation is fused with the mean neighborhood representation.
5. A classifier predicts the center-pixel label.

The pretrained spectral encoder can remain frozen, preserving its wavelength-aware representation while training only the spatial head.

## First experiment

Run:

```bash
bash scripts/z_submit_local_probe_indian.sh
```

Equivalent core arguments:

```bash
python hypersl_linear_probe.py \
  --checkpoint /path/to/checkpoint.pt \
  --data-path /path/to/IndianPine.mat \
  --embedding-dim 128 \
  --encoder-depth 8 \
  --decoder-depth 8 \
  --num-heads 8 \
  --head-type local_attention \
  --freeze-encoder \
  --patch-size 7 \
  --spatial-depth 2 \
  --spatial-heads 4 \
  --batch-size 4 \
  --test-batch-size 16 \
  --grad-accum-steps 8 \
  --lr 3e-4
```

After the frozen-head experiment converges, remove `--freeze-encoder`, add `--encoder-lr 1e-5`, and reduce the head learning rate to approximately `1e-4`.

## Comparisons to run

Use the same data split and seed for all three:

```text
head=linear, patch=1, frozen encoder
head=cnn, patch=7, frozen encoder
head=local_attention, patch=7, frozen encoder
```

Then compare OA, AA, Kappa, parameter count, and epoch time.

## Evaluation warning

Random pixel splits combined with overlapping `7x7` patches can put nearly identical neighborhoods in training and testing. This can substantially inflate classification accuracy. For a defensible locality experiment, use spatially separated train/test regions or remove test centers within the patch radius of any training center.

## Other corrections included

- Corrected the last Indian Pines removed-band index from zero-based `119` to `219`, corresponding to band 220.
- Removed hard-coded W&B keys from the patched copies. Authenticate with `wandb login` or the `WANDB_API_KEY` environment variable instead.

The exposed keys should be revoked and replaced because deleting them from the current files does not remove them from repository history or previously shared archives.
