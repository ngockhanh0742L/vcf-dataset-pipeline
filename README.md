# VCF Dataset Pipeline: Deepfake Detection Data Preparation Framework

## 1. Abstract and System Overview

This repository provides a highly optimized Extract, Transform, Load (ETL) pipeline exclusively designed for **Deepfake Detection Dataset Preparation**. It transforms raw video datasets into high-quality, motion-aware face sequences tailored for training advanced spatiotemporal neural network architectures (like EfficientNet-B3 + BiLSTM, 3D-CNNs, etc.).

Unlike naive frame extraction pipelines, this framework employs a rigorous frame-selection algorithm based on structural similarity (SSIM) and Laplacian variance. This ensures that any deep learning model trained on this dataset learns from dynamic, high-fidelity facial movements rather than static or blurry artifacts. Furthermore, all intermediate face cropping operations are performed entirely in memory (RAM), vastly accelerating processing speeds by minimizing disk I/O bottlenecks.

## 2. System Architecture and Data Flow

The preprocessing pipeline acts as a sophisticated filter funnel, transforming raw `.mp4` videos into tightly structured `24-frame` sequence tensors. The internal flow is defined as follows:

### 2.1. Temporal Sampling (`src/sampler.py`)
- **Timestamp-Based Extraction**: Videos are read using OpenCV. Instead of extracting all frames blindly, the system samples frames based on precise timestamp intervals.
- **Candidate vs. Model FPS**: The pipeline samples a dense pool of "Candidate Frames" at `candidate_fps = 15.0`. Later, the sequence builder sub-samples these down to the target `model_fps = 7.5`. This provides a temporal buffer, allowing the system to shift selections slightly if a target frame is heavily blurred or missing a face.

### 2.2. Face Detection and Alignment (`src/face_processor.py`)
- **Engine**: Utilizes **MediaPipe Face Detection** (BlazeFace backend) operating in `min_detection_confidence=0.5` mode.
- **Cropping & Sizing**: Faces are detected, and a bounding box is established. A **30% margin** (factor of 1.3) is added to include the entire head structure (chin, forehead, and edges of the face) which is crucial for identifying artifact boundaries in Deepfakes.
- The crop is subsequently resized to `300x300` (RGB) to standardize the input resolution.

### 2.3. Quality Assessment (`src/quality.py`)
- Uses the **Variance of the Laplacian** operator on the grayscale representation of the cropped face.
- Frames returning a variance below `filter.blur_threshold` are flagged as invalid (too blurry) and excluded from the selection pool.

### 2.4. Sequence Building & Motion Analysis (`src/sequence_builder.py` & `src/motion.py`)
- **Sliding Window**: Extracts sequences of exactly `seq_len = 24` frames, utilizing a `window_hop` to define the overlap between consecutive sequences.
- **SSIM Motion Filtering**: Calculates the Structural Similarity Index (SSIM) between consecutive selected frames. A low standard deviation in SSIM indicates a frozen/static video (often an artifact of poor deepfakes or dead data). Sequences failing `filter.motion_std_threshold` are discarded.
- **Repair Policy**: If an occlusion occurs and a face is missing for a single frame, the system intelligently duplicates the nearest valid frame. Sequences exceeding the `max_missing_faces` threshold are permanently discarded to maintain data integrity.

## 3. Directory Structure

```text
vcf-dataset-pipeline/
├── config.yaml             # Core configuration (Hyperparameters, IO Paths)
├── pipeline.py             # Main CLI Entry Point
├── data/
│   ├── raw_videos/         # Source input directories (e.g., /real, /fake MP4s)
│   ├── sequences/          # Generated dataset tensors (300x300 JPEG sequences)
│   └── manifests/          # sequence_manifest.csv (Ground truth & metadata)
└── src/
    ├── config.py           # YAML parser and validator
    ├── face_processor.py   # MediaPipe face extraction
    ├── manifest.py         # CSV Logger & Writer
    ├── motion.py           # SSIM algorithms
    ├── quality.py          # Laplacian algorithms
    ├── sampler.py          # Timestamp interpolation
    ├── selector.py         # Sequence ranking algorithm
    ├── sequence_builder.py # Tensor assembly and repair policies
    ├── utils.py            # Logging and path utilities
    └── video_reader.py     # cv2.VideoCapture wrapper
```

## 4. Dataset Manifest Format

The pipeline outputs `sequence_manifest.csv` which acts as the master ground truth database for your training pipeline. Each row represents a 24-frame sequence:
- `sequence_id`: Unique UUID for the tensor sequence.
- `video_id` / `label` / `split`: Source identifier and classification target.
- `frame_paths`: JSON array pointing to the specific JPEG crops inside the `sequences/` directory.
- `avg_quality` / `avg_motion` / `repair_ratio`: Granular metadata allowing for dynamic filtering during your own dataloader phase (e.g. via `tf.data.Dataset` or PyTorch `DataLoader`).

## 5. Execution Commands

The interface is managed via `pipeline.py`. Ensure the `mediapipe_env` conda environment is active.

### Run Preprocessing (Dataset Generation)
Extracts faces, runs filters, builds sequences, and writes to `data/sequences`.
```bash
python pipeline.py --mode preprocess
```
*(Note: Since this repository acts strictly as a dataset ETL pipeline, training and evaluation logic are decoupled and left for your downstream ML environment).*
