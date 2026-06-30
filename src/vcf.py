"""VCF-specific path parsing and leakage-safe deterministic splits."""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path


COMPRESSIONS = {"raw", "c23", "c40"}
GEN_METHODS = {
    "deeplivecam",
    "deeplivecam_enhance",
    "simswap_224",
    "simswap_512",
    "targets",
}


@dataclass(frozen=True)
class VideoSample:
    path: str
    video_id: str
    compression: str
    gen_method: str
    resolution: str
    background_type: str
    video_name: str
    group_id: str
    label: int
    class_name: str
    split: str
    dataset_id: str = "vcf"
    media_type: str = ""

    def to_dict(self):
        return asdict(self)


def _find_layout_start(parts):
    for index, part in enumerate(parts[:-1]):
        if (
            part.lower() in COMPRESSIONS
            and index + 1 < len(parts)
            and parts[index + 1].lower() in GEN_METHODS
        ):
            return index
    return None


def assign_split(group_id, config_split):
    ratios = (
        float(config_split.train_ratio),
        float(config_split.val_ratio),
        float(config_split.test_ratio),
    )
    if any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("split ratios must be non-negative and sum to 1.0")

    digest = hashlib.sha256(f"{config_split.seed}:{group_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < ratios[0]:
        return "train"
    if value < ratios[0] + ratios[1]:
        return "val"
    return "test"


def parse_vcf_video(video_path, raw_video_dir, config_split, dataset_id="vcf"):
    if getattr(config_split, "group_by", "background_and_video_name") != "background_and_video_name":
        raise ValueError("Only split.group_by=background_and_video_name is supported")
    raw_root = Path(raw_video_dir).resolve()
    path = Path(video_path).resolve()
    try:
        relative = path.relative_to(raw_root)
    except ValueError as exc:
        raise ValueError(f"Video is outside raw_video_dir: {path}") from exc

    parts = relative.parts
    start = _find_layout_start(parts)
    tail_length = 0 if start is None else len(parts) - start
    if start is None or tail_length not in {5, 6}:
        raise ValueError(
            "Expected VCF path compression/gen_method/resolution/"
            "[media_type/]background_type/video.mp4, got: "
            f"{relative.as_posix()}"
        )

    compression, gen_method, resolution = (
        part.lower() for part in parts[start : start + 3]
    )
    if tail_length == 6:
        media_type = parts[start + 3].lower()
        background_type = parts[start + 4].lower()
    else:
        media_type = ""
        background_type = parts[start + 3].lower()
    if compression not in COMPRESSIONS:
        raise ValueError(f"Unsupported VCF compression '{compression}'")
    if gen_method not in GEN_METHODS:
        raise ValueError(f"Unsupported VCF gen_method '{gen_method}'")

    video_name = path.stem
    group_id = f"{dataset_id}:{background_type}/{video_name}".lower()
    label = 0 if gen_method == "targets" else 1
    return VideoSample(
        path=str(path),
        video_id=relative.as_posix(),
        compression=compression,
        gen_method=gen_method,
        resolution=resolution,
        background_type=background_type,
        video_name=video_name,
        group_id=group_id,
        label=label,
        class_name="real" if label == 0 else "fake",
        split=assign_split(group_id, config_split),
        dataset_id=dataset_id,
        media_type=media_type,
    )


# Backward-compatible import name for downstream code written for v2.
VCFVideo = VideoSample
