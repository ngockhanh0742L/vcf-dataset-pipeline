import hashlib
import json
from pathlib import Path

import yaml


class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, _convert(value))

    def __repr__(self):
        return str(self.__dict__)

    def to_dict(self):
        return {key: _plain(value) for key, value in self.__dict__.items()}


def _convert(value):
    if isinstance(value, dict):
        return Config(value)
    if isinstance(value, list):
        return [_convert(item) for item in value]
    return value


def _plain(value):
    if isinstance(value, Config):
        return value.to_dict()
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _resolve_path(base_dir, value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _require(config, section, names):
    target = getattr(config, section, None)
    if target is None:
        raise ValueError(f"Missing config section: {section}")
    missing = [name for name in names if not hasattr(target, name)]
    if missing:
        raise ValueError(f"Missing config key(s) in {section}: {', '.join(missing)}")


def validate_config(config):
    _require(
        config,
        "data",
        [
            "output_dir",
            "sequence_dir",
            "manifest_path",
            "split_manifest_path",
            "summary_path",
            "output_format",
            "fail_on_video_error",
        ],
    )
    _require(
        config,
        "pipeline",
        ["candidate_fps", "model_fps", "seq_len", "window_hop", "image_size"],
    )
    _require(config, "split", ["seed", "train_ratio", "val_ratio", "test_ratio"])
    if not getattr(config, "datasets", None):
        raise ValueError("At least one entry is required in datasets")

    dataset_ids = []
    for source in config.datasets:
        for name in ("id", "adapter", "root"):
            if not hasattr(source, name):
                raise ValueError(f"Every dataset requires '{name}'")
        dataset_id = str(source.id).strip().lower()
        if not dataset_id or any(char in dataset_id for char in "/\\:"):
            raise ValueError(f"Invalid dataset id: {source.id!r}")
        dataset_ids.append(dataset_id)
        source.id = dataset_id
        source.adapter = str(source.adapter).strip().lower()
        if source.adapter not in {"vcf", "manifest"}:
            raise ValueError(f"Unsupported dataset adapter: {source.adapter}")
        if source.adapter == "manifest" and not hasattr(source, "manifest"):
            raise ValueError(f"Manifest dataset '{source.id}' requires a manifest path")
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("Dataset ids must be unique")

    candidate_fps = float(config.pipeline.candidate_fps)
    model_fps = float(config.pipeline.model_fps)
    if candidate_fps <= 0 or model_fps <= 0:
        raise ValueError("candidate_fps and model_fps must be positive")
    if model_fps > candidate_fps:
        raise ValueError("model_fps cannot exceed candidate_fps")
    for name in ("seq_len", "window_hop", "image_size"):
        if int(getattr(config.pipeline, name)) <= 0:
            raise ValueError(f"pipeline.{name} must be positive")
    if int(getattr(config.pipeline, "checkpoint_every_videos", 25)) <= 0:
        raise ValueError("pipeline.checkpoint_every_videos must be positive")
    if int(getattr(config.pipeline, "workers", 1)) <= 0:
        raise ValueError("pipeline.workers must be positive")

    ratios = [
        float(config.split.train_ratio),
        float(config.split.val_ratio),
        float(config.split.test_ratio),
    ]
    if any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("split ratios must be non-negative and sum to 1.0")
    if config.data.output_format.lower().lstrip(".") not in {"png", "jpg", "jpeg"}:
        raise ValueError("data.output_format must be png, jpg, or jpeg")
    jpeg_quality = int(getattr(config.data, "jpeg_quality", 95))
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("data.jpeg_quality must be between 1 and 100")

    output_root = Path(config.data.output_dir)
    for name in ("sequence_dir", "manifest_path", "split_manifest_path", "summary_path"):
        path = Path(getattr(config.data, name))
        try:
            path.relative_to(output_root)
        except ValueError as exc:
            raise ValueError(f"data.{name} must be inside data.output_dir") from exc


def load_config(config_path="config.yaml"):
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config_dict = yaml.safe_load(handle)
    if not isinstance(config_dict, dict):
        raise ValueError("Config root must be a YAML mapping")

    config = Config(config_dict)
    base_dir = path.parent
    for name in ("output_dir", "sequence_dir", "manifest_path", "split_manifest_path", "summary_path"):
        if hasattr(config.data, name):
            setattr(config.data, name, _resolve_path(base_dir, getattr(config.data, name)))
    for source in getattr(config, "datasets", []):
        source.root = _resolve_path(base_dir, source.root)
        if hasattr(source, "manifest"):
            source.manifest = _resolve_path(base_dir, source.manifest)
    config.config_path = str(path)
    validate_config(config)
    return config


def pipeline_fingerprint(config):
    semantic_pipeline_keys = (
        "candidate_fps",
        "model_fps",
        "seq_len",
        "window_hop",
        "image_size",
        "max_sequences_per_video",
        "use_timestamp_sampling",
    )
    payload = {
        "pipeline": {
            key: getattr(config.pipeline, key) for key in semantic_pipeline_keys
        },
        "face": config.face.to_dict(),
        "quality": config.quality.to_dict(),
        "motion": config.motion.to_dict(),
        "split": config.split.to_dict(),
        "output_format": config.data.output_format,
    }
    if config.data.output_format.lower().lstrip(".") in {"jpg", "jpeg"}:
        payload["jpeg_quality"] = int(getattr(config.data, "jpeg_quality", 95))
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
