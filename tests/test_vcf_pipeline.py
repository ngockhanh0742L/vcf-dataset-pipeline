import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.datasets import _discover_manifest
from src.face_processor import FaceRecord
from src.manifest import write_manifest
from src.quality import compute_quality_score
from src.sequence_builder import build_sequences
from src.validate_output import validate_output
from src.vcf import parse_vcf_video


def namespace(**kwargs):
    return SimpleNamespace(**kwargs)


class VCFMetadataTests(unittest.TestCase):
    def setUp(self):
        self.split = namespace(
            seed=42,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            group_by="background_and_video_name",
        )

    def test_exact_gen_method_controls_label_and_variants_share_split(self):
        root = Path("dataset")
        real = parse_vcf_video(
            root / "raw/targets/270x480/th/contains_target.mp4", root, self.split
        )
        fake = parse_vcf_video(
            root / "c40/simswap_224/1080x1920/th/contains_target.mp4",
            root,
            self.split,
        )
        self.assertEqual(real.label, 0)
        self.assertEqual(fake.label, 1)
        self.assertEqual(real.group_id, fake.group_id)
        self.assertEqual(real.split, fake.split)

    def test_wrapper_directory_is_allowed(self):
        metadata = parse_vcf_video(
            Path("dataset/vcf/c23/deeplivecam/360x640/th-bb/a.mp4"),
            Path("dataset"),
            self.split,
        )
        self.assertEqual(metadata.compression, "c23")
        self.assertEqual(metadata.video_id, "vcf/c23/deeplivecam/360x640/th-bb/a.mp4")

    def test_optional_media_type_directory_is_supported(self):
        metadata = parse_vcf_video(
            Path("dataset/raw/targets/540x960/mp4/th-ob/a.mp4"),
            Path("dataset"),
            self.split,
        )
        self.assertEqual(metadata.media_type, "mp4")
        self.assertEqual(metadata.background_type, "th-ob")

    def test_dataset_id_namespaces_group_and_split(self):
        root = Path("dataset")
        first = parse_vcf_video(
            root / "raw/targets/270x480/th/a.mp4", root, self.split, "first"
        )
        second = parse_vcf_video(
            root / "raw/targets/270x480/th/a.mp4", root, self.split, "second"
        )
        self.assertNotEqual(first.group_id, second.group_id)
        self.assertEqual(first.dataset_id, "first")


class FilteringTests(unittest.TestCase):
    def test_vcf_quality_is_scored_but_not_rejected_by_default(self):
        record = FaceRecord("video", 0, 0.0, 0)
        record.face_image = np.full((32, 32, 3), 128, dtype=np.uint8)
        record.detect_confidence = 0.9
        config = namespace(
            enforce_thresholds=False,
            min_blur=60,
            min_brightness=35,
            max_brightness=225,
            min_contrast=10,
        )
        self.assertTrue(compute_quality_score(record, config))
        self.assertFalse(record.hard_invalid)

    def test_sequence_with_too_many_repairs_is_rejected(self):
        frames = []
        for index in range(4):
            record = FaceRecord("video", index, index / 2, index)
            record.face_image = np.zeros((8, 8, 3), dtype=np.uint8)
            record.quality_score = 0.8
            record.motion_score = 0.1
            record.ssim_to_prev_selected = 0.9 if index else None
            record.repair_flag = index > 0
            frames.append(record)
        pipeline = namespace(seq_len=4, window_hop=1, max_sequences_per_video=6)
        motion = namespace(
            ssim_redundant_threshold=0.995,
            max_missing_face_ratio=0.0,
            max_repair_ratio=0.1,
            max_duplicate_ratio=0.5,
            min_motion_score=0.001,
        )
        self.assertEqual(build_sequences(frames, pipeline, motion), [])


class DatasetAdapterTests(unittest.TestCase):
    def test_generic_manifest_creates_namespaced_sample(self):
        split = namespace(
            seed=42,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            group_by="background_and_video_name",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "clip.mp4").write_bytes(b"")
            manifest = root / "videos.csv"
            pd.DataFrame(
                [{"path": "clip.mp4", "label": 1, "group_id": "subject-1"}]
            ).to_csv(manifest, index=False)
            source = namespace(
                id="custom",
                root=str(root),
                manifest=str(manifest),
                strict_layout=True,
            )
            samples, rejected = _discover_manifest(source, split)
            self.assertEqual(rejected, [])
            self.assertEqual(samples[0].dataset_id, "custom")
            self.assertEqual(samples[0].group_id, "custom:subject-1")
            self.assertEqual(samples[0].label, 1)


class ManifestTests(unittest.TestCase):
    def test_manifest_upsert_preserves_previous_rows(self):
        first = {
            "sequence_id": "stable-id",
            "dataset_id": "vcf",
            "video_id": "raw/targets/270x480/th/a.mp4",
            "label": 0,
            "split": "train",
        }
        second = {
            "sequence_id": "new-id",
            "dataset_id": "other",
            "video_id": "fake/b.mp4",
            "label": 1,
            "split": "val",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "manifest.csv")
            write_manifest([first], path)
            result = write_manifest([second], path)
            self.assertEqual(set(result["sequence_id"]), {"stable-id", "new-id"})


class OutputValidationTests(unittest.TestCase):
    def test_valid_manifest_and_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence_dir = root / "sequences"
            folder = sequence_dir / "sequence-a"
            folder.mkdir(parents=True)
            (folder / "frame_00.png").write_bytes(b"not-decoded-by-validator")
            row = {
                "sequence_id": "sequence-a",
                "dataset_id": "vcf",
                "video_id": "a.mp4",
                "group_id": "vcf:bg/a",
                "label": 0,
                "split": "train",
                "frame_paths": json.dumps(["sequence-a/frame_00.png"]),
                "seq_len": 1,
                "pipeline_fingerprint": "abc",
            }
            manifest = root / "manifest.csv"
            write_manifest([row], str(manifest), preserve_existing=False)
            result = validate_output(manifest, sequence_dir, require_all_splits=False)
            self.assertEqual(result["sequences"], 1)


if __name__ == "__main__":
    unittest.main()
