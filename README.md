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

## 6. Configuration & Hyperparameter Tuning Guide

The dataset preprocessing can be heavily customized by modifying `config.yaml`. Here are key recommended tuning configurations based on your objectives:

### 6.1. Maximizing Dataset Size (Generating More Sequences)
If you want to yield more sequences per raw video (useful for mitigating overfitting):
* **`pipeline.window_hop_train`**: Decrease this value (e.g., from `6` to `3` or `4`). Shorter hop sizes increase the overlap of the sliding windows, yielding more sequences from the same video length.
* **`pipeline.max_sequences_per_video`**: Increase this limit (e.g., from `6` to `10` or `12`) to allow more window extractions from longer source videos.

### 6.2. Relaxing Filtering Rules (Retaining More Quality-Challenged Videos)
If too many videos are getting discarded due to recording conditions:
* **`quality.min_blur`**: Lower this threshold (e.g., from `60.0` to `40.0` or `30.0`) to accept softer, slightly blurrier face crops.
* **`face.min_face_box_ratio`**: Lower this threshold (e.g., from `0.08` to `0.05`) to allow face extraction when subjects are positioned further from the camera.

### 6.3. Strict Motion Filtering (Excluding Static Frames)
If you want to ensure the neural network only trains on dynamic expressions (discarding segments where the person is completely still):
* **`motion.min_motion_score`**: Increase this value (e.g., from `0.005` to `0.01`) to demand higher variation between selected frames.
* **`motion.ssim_redundant_threshold`**: Decrease this threshold (e.g., from `0.995` to `0.990`) to strictly reject near-identical adjacent frames.

### 6.4. Changing Sequence Length & Frame Frequency
* **`pipeline.seq_len`**: Adjust the number of frames per sequence (default: `24`). Decrease to `16` to speed up training and save GPU VRAM, or increase to `32` to capture longer temporal context.
* **`pipeline.model_fps`**: Change the sampling rate. Keeping it at `7.5` means a `24-frame` sequence captures `24 / 7.5 = 3.2` seconds of real-time video. Increasing it captures faster motion dynamics.
