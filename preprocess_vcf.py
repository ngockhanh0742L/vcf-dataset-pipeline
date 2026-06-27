import argparse
import concurrent.futures
import hashlib
import json
import logging
import os
import yaml
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import cv2
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class VideoMetadata:
    path: str
    relative_path: str
    filename: str
    stem: str
    group_id: str
    compression: str
    method: str
    resolution: str
    background: str
    label: int
    label_name: str
    readable: bool = False
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frames: int = 0
    duration_sec: float = 0.0


@dataclass
class CropMetadata:
    crop_path: str
    source_video: str
    relative_path: str
    frame_idx: int
    split: str
    label: int
    label_name: str
    compression: str
    method: str
    resolution: str
    background: str
    group_id: str


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def analyze_video(v_path: Path, root_dir: Path) -> VideoMetadata:
    """Parse a VCF path and probe video to extract its attributes and metadata."""
    rel_path = v_path.relative_to(root_dir)
    parts = rel_path.parts
    
    compression = "unknown"
    method = "unknown"
    resolution = "unknown"
    background = "unknown"
    
    compressions = {"raw", "c23", "c40"}
    methods = {
        "targets", "target", "real", "original", "authentic",
        "deeplivecam", "deeplivecam_enhance", "simswap_224", "simswap_512"
    }
    backgrounds = {"th", "th-bb", "th-m", "th-ob"}
    
    for part in parts:
        part_lower = part.lower()
        if part_lower in compressions:
            compression = part_lower
        elif part_lower in methods:
            method = part_lower
        elif "x" in part_lower:
            res_parts = part_lower.split("x")
            if len(res_parts) == 2 and res_parts[0].isdigit() and res_parts[1].isdigit():
                resolution = part_lower
        elif part_lower in backgrounds:
            background = part_lower
            
    label = 0 if method in {"targets", "target", "real", "original", "authentic"} else 1
    label_name = "real" if label == 0 else "fake"
    
    meta = VideoMetadata(
        path=str(v_path),
        relative_path=str(rel_path),
        filename=v_path.name,
        stem=v_path.stem,
        group_id=v_path.stem,
        compression=compression,
        method=method,
        resolution=resolution,
        background=background,
        label=label,
        label_name=label_name
    )
    
    cap = cv2.VideoCapture(str(v_path))
    if cap.isOpened():
        meta.readable = True
        meta.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        meta.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        meta.fps = cap.get(cv2.CAP_PROP_FPS)
        meta.frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if meta.fps > 0:
            meta.duration_sec = meta.frames / meta.fps
        cap.release()
        
    return meta


def extract_faces_task(kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Multiprocessing task for extracting faces from a single video."""
    video_path = kwargs["video_path"]
    rel_path = kwargs["rel_path"]
    output_dir = kwargs["output_dir"]
    split = kwargs["split"]
    label_name = kwargs["label_name"]
    metadata = kwargs["metadata"]
    max_frames = kwargs["max_frames"]
    face_size = kwargs["face_size"]
    jpeg_quality = kwargs["jpeg_quality"]
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
        
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frames_count <= 0:
        cap.release()
        return []
        
    # Chiến lược trích xuất mới: Lấy 3 chunks (Đầu, Giữa, Cuối), mỗi chunk n frames.
    # Phục vụ cho cả lấy mẫu "rải đều" và lấy mẫu "temporal chunks".
    num_chunks = 3
    frames_per_chunk = max(1, max_frames // num_chunks)
    
    frame_indices = []
    if frames_count <= max_frames:
        frame_indices = list(range(frames_count))
    else:
        step_between_chunks = max(1, (frames_count - frames_per_chunk) // max(1, num_chunks - 1))
        for i in range(num_chunks):
            start_idx = i * step_between_chunks
            # Đảm bảo không vượt quá frames_count
            start_idx = min(start_idx, max(0, frames_count - frames_per_chunk))
            chunk_indices = list(range(start_idx, start_idx + frames_per_chunk))
            frame_indices.extend(chunk_indices)
            
    frame_indices = sorted(list(set(frame_indices)))[:max_frames]
    
    crops_info = []
    path_hash = hashlib.md5(rel_path.encode("utf-8")).hexdigest()[:8]
    
    has_mediapipe = False
    try:
        import mediapipe as mp
        has_mediapipe = True
    except ImportError:
        pass
        
    if has_mediapipe:
        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(image_rgb)
                
                if not results.detections:
                    continue
                    
                largest_face = None
                max_area = 0
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    area = bbox.width * bbox.height
                    if area > max_area:
                        max_area = area
                        largest_face = bbox
                        
                if not largest_face:
                    continue
                    
                h, w, _ = frame.shape
                box_x = int(largest_face.xmin * w)
                box_y = int(largest_face.ymin * h)
                box_w = int(largest_face.width * w)
                box_h = int(largest_face.height * h)
                
                margin_x = int(box_w * 0.35) # Margin 35%
                margin_y = int(box_h * 0.35)
                
                x1 = max(0, box_x - margin_x)
                y1 = max(0, box_y - margin_y)
                x2 = min(w, box_x + box_w + margin_x)
                y2 = min(h, box_y + box_h + margin_y)
                
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue
                    
                face_crop_resized = cv2.resize(face_crop, (face_size, face_size))
                crop_filename = f"{metadata['stem']}_{path_hash}_f{idx:04d}.jpg"
                crop_dir = output_dir / "face_cache" / split / label_name
                crop_dir.mkdir(parents=True, exist_ok=True)
                crop_path = crop_dir / crop_filename
                
                cv2.imwrite(str(crop_path), face_crop_resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                
                crop_meta = CropMetadata(
                    crop_path=str(crop_path.relative_to(output_dir)),
                    source_video=video_path.name,
                    relative_path=rel_path,
                    frame_idx=idx,
                    split=split,
                    label=metadata["label"],
                    label_name=label_name,
                    compression=metadata["compression"],
                    method=metadata["method"],
                    resolution=metadata["resolution"],
                    background=metadata["background"],
                    group_id=metadata["group_id"]
                )
                crops_info.append(asdict(crop_meta))
    else:
        face_cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) == 0:
                continue
                
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            margin_x = int(w * 0.35)
            margin_y = int(h * 0.35)
            
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(frame.shape[1], x + w + margin_x)
            y2 = min(frame.shape[0], y + h + margin_y)
            
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
                
            face_crop_resized = cv2.resize(face_crop, (face_size, face_size))
            crop_filename = f"{metadata['stem']}_{path_hash}_f{idx:04d}.jpg"
            crop_dir = output_dir / "face_cache" / split / label_name
            crop_dir.mkdir(parents=True, exist_ok=True)
            crop_path = crop_dir / crop_filename
            
            cv2.imwrite(str(crop_path), face_crop_resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            
            crop_meta = CropMetadata(
                crop_path=str(crop_path.relative_to(output_dir)),
                source_video=video_path.name,
                relative_path=rel_path,
                frame_idx=idx,
                split=split,
                label=metadata["label"],
                label_name=label_name,
                compression=metadata["compression"],
                method=metadata["method"],
                resolution=metadata["resolution"],
                background=metadata["background"],
                group_id=metadata["group_id"]
            )
            crops_info.append(asdict(crop_meta))
            
    cap.release()
    return crops_info


def main():
    parser = argparse.ArgumentParser(description="VCF Data Preprocessing Pipeline")
    parser.add_argument("--root", type=str, required=True, help="Root directory containing VCF videos")
    parser.add_argument("--output", type=str, required=True, help="Output directory for manifests and crops")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    
    parser.add_argument("--extract-faces", type=str, choices=["true", "false"], help="Enable face extraction")
    parser.add_argument("--max-frames-per-video", type=int, help="Max frames to sample per video")
    parser.add_argument("--face-size", type=int, help="Output face size")
    parser.add_argument("--jpeg-quality", type=int, help="JPEG quality for face crops")
    parser.add_argument("--workers", type=int, help="Number of worker processes")
    parser.add_argument("--debug-n", type=int, help="Number of videos to process in debug mode")
    
    args = parser.parse_args()
    
    root_dir = Path(args.root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = load_config(Path(args.config))
    
    def get_param(name: str, args_val: Any, default_val: Any) -> Any:
        if args_val is not None:
            if isinstance(args_val, str) and args_val.lower() in {"true", "false"}:
                return args_val.lower() == "true"
            return args_val
        if name in config:
            return config[name]
        return default_val
        
    extract_faces = get_param("extract_faces", args.extract_faces, False)
    max_frames_per_video = get_param("max_frames_per_video", args.max_frames_per_video, 18) # Lấy 18 frame (3 chunks x 6 frames)
    face_size = get_param("face_size", args.face_size, 300) # Chuẩn EfficientNet-B3
    jpeg_quality = get_param("jpeg_quality", args.jpeg_quality, 95) # Nâng chất lượng JPEG lên 95 để giữ artifact
    workers = get_param("workers", args.workers, 1)
    debug_n = get_param("debug_n", args.debug_n, None)
    
    logging.info("Starting VCF pipeline")
    logging.info(f"Root: {root_dir}")
    logging.info(f"Output: {output_dir}")
    logging.info(f"Extract faces: {extract_faces} (workers: {workers})")
    
    if not root_dir.exists():
        logging.error("Root directory does not exist.")
        return
        
    video_files = []
    extensions = {".mp4", ".avi", ".mkv", ".mov"}
    for ext in extensions:
        video_files.extend(list(root_dir.rglob(f"*{ext}")))
        
    if debug_n and debug_n > 0:
        video_files = video_files[:debug_n]
        logging.info(f"Debug mode enabled. Limited to {len(video_files)} videos.")
        
    logging.info(f"Found {len(video_files)} videos. Parsing and probing...")
    
    manifest_rows = []
    unreadable_videos = []
    bad_paths = []
    
    for i, v_path in enumerate(tqdm(video_files, desc="Probing videos")):
        meta = analyze_video(v_path, root_dir)
        
        if meta.compression == "unknown" and meta.method == "unknown":
            bad_paths.append(meta.relative_path)
            
        if not meta.readable:
            unreadable_videos.append(meta.relative_path)
            
        manifest_rows.append(asdict(meta))
        
        if (i + 1) % 50 == 0:
            logging.info(f"Processed {i + 1} / {len(video_files)} videos.")
            
    if bad_paths:
        logging.warning(f"Found {len(bad_paths)} potentially bad paths.")
        for bp in bad_paths[:20]:
            logging.warning(f"Bad path example: {bp}")
            
    if unreadable_videos:
        logging.warning(f"Found {len(unreadable_videos)} unreadable videos.")
        for uv in unreadable_videos[:20]:
            logging.warning(f"Unreadable video example: {uv}")
            
    if not manifest_rows:
        logging.warning("No videos processed. Exiting.")
        return
        
    df = pd.DataFrame(manifest_rows)
    df.to_csv(output_dir / "vcf_manifest.csv", index=False)
    
    logging.info("Creating splits based on group_id")
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    
    if len(df) > 1:
        train_idx, temp_idx = next(gss.split(df, groups=df["group_id"]))
        train_df = df.iloc[train_idx].copy()
        temp_df = df.iloc[temp_idx].copy()
        
        if len(temp_df) > 1:
            gss_val = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
            val_idx, test_idx = next(gss_val.split(temp_df, groups=temp_df["group_id"]))
            val_df = temp_df.iloc[val_idx].copy()
            test_df = temp_df.iloc[test_idx].copy()
        else:
            val_df = temp_df.copy()
            test_df = pd.DataFrame(columns=temp_df.columns)
    else:
        train_df = df.copy()
        val_df = pd.DataFrame(columns=df.columns)
        test_df = pd.DataFrame(columns=df.columns)
        
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    
    splits_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    splits_df.to_csv(output_dir / "vcf_manifest_splits.csv", index=False)
    
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    
    summary = {
        "total_videos": len(df),
        "readable": int(df["readable"].sum()) if "readable" in df else 0,
        "splits": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df)
        },
        "labels": df["label_name"].value_counts().to_dict(),
        "compressions": df["compression"].value_counts().to_dict(),
        "methods": df["method"].value_counts().to_dict(),
        "resolutions": df["resolution"].value_counts().to_dict(),
        "backgrounds": df["background"].value_counts().to_dict()
    }
    
    if extract_faces:
        logging.info("Starting face extraction...")
        crops_manifest_rows = []
        extraction_tasks = []
        
        for idx, row in splits_df.iterrows():
            if not row["readable"]:
                continue
                
            extraction_tasks.append({
                "video_path": root_dir / row["relative_path"],
                "rel_path": row["relative_path"],
                "output_dir": output_dir,
                "split": row["split"],
                "label_name": row["label_name"],
                "metadata": row.to_dict(),
                "max_frames": max_frames_per_video,
                "face_size": face_size,
                "jpeg_quality": jpeg_quality
            })
            
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(extract_faces_task, task): task for task in extraction_tasks}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Extracting faces"):
                try:
                    res = future.result()
                    if res:
                        crops_manifest_rows.extend(res)
                except Exception as e:
                    logging.error(f"Error extracting faces: {e}")
                    
        crops_df = pd.DataFrame(crops_manifest_rows)
        manifest_name = "face_cache_debug_manifest.csv" if (debug_n and debug_n > 0) else "face_cache_manifest.csv"
        
        if not crops_df.empty:
            crops_df.to_csv(output_dir / manifest_name, index=False)
            summary["face_crops_extracted"] = len(crops_df)
            logging.info(f"Extracted {len(crops_df)} face crops.")
        else:
            summary["face_crops_extracted"] = 0
            logging.info("No face crops were extracted.")
        
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
    logging.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
