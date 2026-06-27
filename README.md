# Data Preprocessing for VCF Dataset

This document provides a comprehensive overview, execution architecture, and operational guidelines for the Data Preprocessing pipeline of the VCF video dataset. This pipeline is responsible for preparing and standardizing training data for Deep Learning neural networks in Deepfake Detection tasks.

---

## 1. System Requirements and Setup

The system requires standard computational and Computer Vision libraries. Deployment within a Virtual Environment is highly recommended to prevent dependency conflicts.

```bash
# Install dependencies
pip install -r requirements.txt

# Highly recommended: Install MediaPipe to optimize the Face Detection algorithm
pip install mediapipe
```

System configuration parameters are managed via configuration files:
* **File:** `configs/default.yaml` - Stores hyperparameters such as `face_size`, `max_frames_per_video`, and `jpeg_quality`.

---

## 2. Preprocessing Methodology and Workflow

The Data Preprocessing workflow is entirely automated. Below is the execution flow, the technical significance of each step, and the corresponding scripts responsible for them.

### 2.1. Metadata Parsing
* **Executable File:** `preprocess_vcf.py`
* **Description:** The system recursively scans the directory structure containing raw videos to extract metadata, including compression format, generation method, spatial resolution, background type, and ground-truth label.
* **Significance:** Gathering metadata is crucial for Exploratory Data Analysis (EDA). It enables the management of data distribution and the mitigation of factors that could introduce model bias.
* **Supplementary Tools:** The script `eda_vcf.py` can be executed independently to visualize statistical distributions derived from the acquired metadata.

### 2.2. Dataset Splitting
* **Executable File:** `preprocess_vcf.py`
* **Description:** The dataset is partitioned into Training, Validation, and Testing splits. The `GroupShuffleSplit` algorithm (from the `scikit-learn` library) is applied based on the `group_id` attribute (the source video identifier).
* **Significance:** This approach effectively prevents Data Leakage by ensuring that face crops generated from the same source video do not cross-contaminate between different splits. This guarantees the Generalization capabilities of the model during evaluation on unseen data.

### 2.3. Face Detection and Alignment
* **Executable File:** `preprocess_vcf.py` (or `resume_preprocess.py` for process recovery)
* **Description:** The system performs Frame Sampling along the duration of the video. At each sampled frame, a Face Detection algorithm (MediaPipe or Haar Cascade) calculates a spatial Bounding Box. A 30 percent margin is then applied to the Bounding Box to encompass the entire head region. The face is cropped from the original frame and interpolated to a standard 224x224 dimension.
* **Significance:** Standardizing the spatial dimensions allows Convolutional Neural Networks (CNNs) or Vision Transformers (ViTs) to focus on extracting features (e.g., texture artifacts) specifically from the facial region without background interference. The 224x224 dimension is fully compatible with the input shape requirements of standard architectures such as ResNet and EfficientNet.

### 2.4. Data Loading (PyTorch Dataset)
* **Executable File:** `dataset.py`
* **Description:** This file provides the `VCFDataset` and `SequenceDeepfakeDataset` classes, which inherit from `torch.utils.data.Dataset`.
* **Significance:** This module is responsible for loading the 224x224 face crops from the local storage into memory, performing Data Augmentation (simulating webcam noise, compression artifacts, etc.), and supplying batch data directly for the Model Training process.

### 2.5. Packaged Inference Preprocessor
* **Executable File:** `inference_preprocessor.py`
* **Description:** This module encapsulates the entire preprocessing logic—Face Detection, Bounding Box margin expansion (30 percent), Resizing (224x224), and Normalization (Albumentations)—into a single, highly cohesive `InferencePreprocessor` class.
* **Significance:** It accepts a raw video frame (numpy array) as input and outputs a normalized PyTorch Tensor (shape `[1, 3, 224, 224]`) ready to be fed directly into the Deep Learning model. This serves as the ideal bridging component between real-time applications (such as a WebRTC server) and the inference model.

---

## 3. Execution Guidelines

### 3.1. Initialize Metadata and Splits (Excluding Face Extraction)
Utilize the main script to parse metadata and partition the data without extracting images. This process outputs CSV Manifest files.
```bash
python preprocess_vcf.py --root /path/to/vcf/dataset --output /path/to/output/dir
```

### 3.2. Comprehensive Pipeline Execution (Face Extraction)
Activate the dataset splitting mechanism and Face Extraction concurrently. Configure the `--workers` parameter to increase Multiprocessing throughput.
```bash
python preprocess_vcf.py --root /path/to/vcf/dataset --output /path/to/output/dir --extract-faces true --workers 4
```

### 3.3. Resume Execution
In the event of a system interruption (such as power failure or process termination), use the recovery script. This script reads the current manifest, bypasses completed data, and continues processing the remaining files.
```bash
# Update root_dir and output_dir paths inside resume_preprocess.py before executing
python resume_preprocess.py
```

### 3.4. Data Output Verification (Sanity Check)
Generate a random matrix of face crops to visually verify the accuracy of the Bounding Box algorithm.
```bash
python inspect_vcf.py --manifest /path/to/output/face_cache_manifest.csv --output preview.jpg
```

---

## 4. Real-time System Integration Guidelines (Web Server Integration)

This Preprocessing module is highly modular, facilitating the direct mapping of its logic into a real-time WebRTC Video Call system (e.g., FastAPI Backend) via the following integration workflow:

1. **Network Topology:** The Web Server acts as a Media Endpoint (SFU/MCU via libraries like `aiortc`), receiving the Video Stream from the Client via the WebRTC protocol.
2. **Frame Extraction:** The Server intercepts the MediaStreamTrack, converting the raw video signal into a Numpy Array at a predetermined frequency (e.g., 3 frames per second).
3. **Pipeline Synchronization (Using `inference_preprocessor.py`):** The Backend integrates the `InferencePreprocessor` class. By passing the extracted frame to the preprocessor, the server replicates the exact Face Detection and Alignment logic used during training. This synchronization is mandatory to maintain a consistent Feature Space between the Training phase and the Inference phase.
4. **Model Inference and Response:** The resulting `[1, 3, 224, 224]` Tensor from the preprocessor is passed directly into the Deepfake Detection Model. The calculated probability (Confidence Score) is returned to the Client via the WebSocket protocol to trigger Frontend actions (e.g., displaying warnings or blocking the stream).
