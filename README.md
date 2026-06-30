# Multi-dataset Face Sequence Pipeline

This project converts labeled videos into leakage-safe face sequences and a
framework-neutral manifest. VCF is supported directly; other datasets can be
added through a small CSV manifest without changing pipeline code.

## Environment

```powershell
conda env create -f environment.yml
conda activate mediapipe_env
```

All paths in `config.yaml` are resolved relative to that config file, so the
commands do not depend on the current working directory.

## Configure input datasets

VCF uses its native layout:

```yaml
datasets:
  - id: vcf
    adapter: vcf
    root: data/vcf
    strict_layout: true
```

Accepted VCF layouts are:

```text
{root}/{compression}/{gen_method}/{resolution}/{background}/{video}.mp4
{root}/{compression}/{gen_method}/{resolution}/{media_type}/{background}/{video}.mp4
{root}/vcf/{compression}/{gen_method}/{resolution}/{background}/{video}.mp4
```

`targets` is real (`0`); the supported generation methods are fake (`1`).
All compression/generator/resolution variants sharing background and filename
receive the same split.

For another dataset, create a CSV with required columns `path`, `label`, and
`group_id`, then add:

```yaml
  - id: another_dataset
    adapter: manifest
    root: D:/datasets/another_dataset
    manifest: D:/datasets/another_dataset/videos.csv
```

`path` may be relative to `root`. `label` must be `0` or `1`. `group_id` must
identify the original subject/source shared by all derived videos; this is the
key that prevents train/validation/test leakage. Optional columns are
`video_id`, `class_name`, `compression`, `gen_method`, `resolution`, `media_type`,
`background_type`, and `video_name`.

## Validate and process

First scan paths, labels, config, and split assignment without writing output:

```powershell
python pipeline.py --validate-only
```

Then process everything, or one named dataset:

```powershell
python pipeline.py --log-file output/preprocess.log
python pipeline.py --dataset vcf
```

Resume is enabled by default. A video is skipped only when it already has an
accepted sequence produced with the same pipeline fingerprint. Use
`--no-resume` to reprocess and upsert, or `--overwrite` for fresh manifests.
The manifest is checkpointed every 25 attempted videos. `--max-videos 1` is
available for a quick end-to-end smoke test.

The checked-in profile uses eight workers. On the reference laptop (Intel Core
Ultra 5 225U, 14 logical processors, 15 GiB RAM), the benchmark was:

| Workers | 8-video time |
|---:|---:|
| 1 | 49.3 s |
| 2 | 32.6 s |
| 4 | 24.4 s |
| 6 | 23.8 s |
| 8 | 21.5 s |

After subtracting dataset discovery, the 9,600-video run is estimated at
roughly 6.5–8 hours. Laptop temperature, video duration, antivirus scanning,
and storage speed can move this estimate. Follow progress separately:

```powershell
Get-Content output/preprocess.log -Wait
```

Validate every manifest reference and leakage group before upload:

```powershell
python -m src.validate_output --config config.yaml
```

The upload check requires non-empty train/validation/test splits and both
labels in train. Add `--allow-partial` only when checking smoke-test output.

Outputs are deliberately isolated from raw videos:

```text
output/
├── sequences/{sequence_id}/frame_00.png ...
└── manifests/
    ├── sequence_manifest.csv
    ├── split_manifest.csv
    └── dataset_summary.json
```

PNG is the lossless research default, so preprocessing does not add another
compression signal to the source dataset. Set `output_format: jpg` with
`jpeg_quality: 95` for a smaller Kaggle-friendly export.

The default keeps six temporally distributed sequences per video. Based on the
included smoke test, the current 9,600-video VCF copy is expected to produce
roughly 144 GiB as PNG. A smaller profile using two JPEG-95 windows per video
would be roughly 10 GiB. Actual compression varies by content.

The hash folder is a stable `sequence_id`, not a class label. Always read
`label`, `class_name`, and `split` from `sequence_manifest.csv`; the training
loader does this automatically. This avoids silently mislabeling a new dataset
whose directory names follow a different convention.

The reference drive had 212 GiB free before the full run. That is enough for
the estimated 144 GiB output, but not for an additional same-size temporary
archive. Use external scratch space or multiple smaller shards for upload.

## Upload to Kaggle

Generate `dataset-metadata.json` only after the output passes validation:

```powershell
python prepare_kaggle.py --owner YOUR_KAGGLE_NAME --slug face-sequences `
  --title "Face Sequences" --license other
kaggle datasets create -p output -r zip
```

Add `--public` to both commands only after confirming that every source
dataset's license permits redistribution. Private is the default.

The official Kaggle CLI and credentials must be installed separately.

## Kaggle TensorFlow baseline

Attach the uploaded dataset to a GPU notebook, upload `kaggle_train.py`, and
run (replace the mounted folder name):

```bash
python kaggle_train.py \
  --data-root /kaggle/input/face-sequences \
  --backbone b0 --batch-size 2 --epochs 10
```

The baseline reads the provided split, applies class weights, freezes an
ImageNet EfficientNet backbone, trains a bidirectional LSTM, and saves the best
model by validation AUC. `--backbone b3` is available but substantially more
memory-intensive.

## Tests

```powershell
python -m unittest discover -s tests -v
```
