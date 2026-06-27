import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SequenceDeepfakeDataset(Dataset):
    """
    Spatio-Temporal Dataset for EfficientNet-B3 + Temporal Aggregation.
    Loads sequences of face crops for each video to support temporal modeling.
    """
    def _split_into_chunks(self, crop_paths, frame_indices):
        if not crop_paths: 
            return []
        chunks = []
        current_chunk = [crop_paths[0]]
        for i in range(1, len(crop_paths)):
            # Nếu frame cách nhau <= 5, coi như cùng 1 chunk liên tục
            if frame_indices[i] - frame_indices[i-1] <= 5:
                current_chunk.append(crop_paths[i])
            else:
                chunks.append(current_chunk)
                current_chunk = [crop_paths[i]]
        chunks.append(current_chunk)
        return chunks

    def __init__(self, manifest_path, base_dir, seq_len=12, split='train', transform=None):
        self.base_dir = base_dir
        self.seq_len = seq_len
        self.split = split
        self.transform = transform
        
        # Load face crops manifest
        df = pd.read_csv(manifest_path)
        
        # Group by video to form sequences
        self.video_groups = []
        for vid, group in df.groupby('source_video'):
            # Sort by frame index to maintain temporal order
            group = group.sort_values('frame_idx')
            crop_paths = group['crop_path'].tolist()
            frame_indices = group['frame_idx'].tolist()
            label = group['label'].iloc[0] # 0 for real, 1 for fake
            
            # Tách thành các chunks dựa trên khoảng cách frame_idx
            chunks = self._split_into_chunks(crop_paths, frame_indices)
            
            self.video_groups.append({
                'video_name': vid,
                'chunks': chunks,
                'label': label
            })

    def __len__(self):
        return len(self.video_groups)

    def __getitem__(self, idx):
        item = self.video_groups[idx]
        label = item['label']
        chunks = item['chunks']
        
        selected_crops = []
        
        if self.split == 'train':
            # Random chọn 2 hoặc 3 chunks
            num_chunks_to_select = np.random.choice([2, 3])
            
            if len(chunks) >= num_chunks_to_select:
                # Chọn random nhưng giữ đúng thứ tự thời gian (đầu -> giữa -> cuối)
                selected_chunk_indices = sorted(np.random.choice(len(chunks), num_chunks_to_select, replace=False))
                selected_chunks = [chunks[i] for i in selected_chunk_indices]
            else:
                selected_chunks = chunks
                num_chunks_to_select = len(selected_chunks)
                
            # Phân bổ số lượng frame cần lấy cho mỗi chunk
            if num_chunks_to_select > 0:
                frames_per_chunk = self.seq_len // num_chunks_to_select
                remainder = self.seq_len % num_chunks_to_select
                
                for i, c in enumerate(selected_chunks):
                    need = frames_per_chunk + (1 if i < remainder else 0)
                    if len(c) >= need:
                        # Trong mỗi chunk, giữ các frame LIÊN TIẾP ngẫu nhiên
                        start = np.random.randint(0, len(c) - need + 1)
                        selected_crops.extend(c[start : start + need])
                    else:
                        # Padding bằng frame cuối nếu chunk ngắn hơn dự kiến
                        selected_crops.extend(c + [c[-1]] * (need - len(c)))
                        
        else:
            # Val/Test: cố định dùng cả 3 chunks (hoặc số chunk tối đa hiện có)
            num_chunks_to_use = min(3, len(chunks))
            if num_chunks_to_use > 0:
                frames_per_chunk = self.seq_len // num_chunks_to_use
                remainder = self.seq_len % num_chunks_to_use
                
                for i in range(num_chunks_to_use):
                    c = chunks[i]
                    need = frames_per_chunk + (1 if i < remainder else 0)
                    if len(c) >= need:
                        # Val/Test cố định lấy phần giữa của mỗi chunk
                        start = max(0, (len(c) - need) // 2)
                        selected_crops.extend(c[start : start + need])
                    else:
                        selected_crops.extend(c + [c[-1]] * (need - len(c)))

        # Fallback padding cực đoan nếu vẫn thiếu frame
        if len(selected_crops) < self.seq_len:
            if len(selected_crops) > 0:
                selected_crops += [selected_crops[-1]] * (self.seq_len - len(selected_crops))
        # Cắt bớt nếu dư frame
        elif len(selected_crops) > self.seq_len:
            selected_crops = selected_crops[:self.seq_len]
            
        frames = []
        for crop_rel_path in selected_crops:
            img_path = os.path.join(self.base_dir, crop_rel_path)
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                # Fallback to zero tensor if corrupted (rare, but safe)
                frames.append(torch.zeros(3, 300, 300))
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            else:
                # Default ToTensor logic if no transform provided
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                
            frames.append(image)
            
        # Stack into [Seq_len, C, H, W]
        sequence_tensor = torch.stack(frames)
        return sequence_tensor, torch.tensor(label, dtype=torch.float32)

def get_web_conference_augmentations(img_size=300):
    """
    Data Augmentations tailored for Real-Time Web Demo conditions.
    Simulates webcam noise, low bandwidth compression, poor lighting, and forces spatial attention.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        
        # Web / Compression artifacts
        A.OneOf([
            A.ImageCompression(quality_lower=30, quality_upper=80, p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.4),
        
        # Webcam Sensor Noise / Lighting
        A.OneOf([
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        ], p=0.4),
        
        # Coarse Dropout (Random Erasing) to force attention away from obvious artifacts (like just eyes/mouth)
        A.CoarseDropout(max_holes=8, max_height=int(img_size*0.1), max_width=int(img_size*0.1), 
                        min_holes=2, min_height=8, min_width=8, p=0.2),
                        
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_balanced_dataloader(dataset, batch_size=16, num_workers=4):
    """
    Creates a DataLoader with WeightedRandomSampler to handle the 4:1 Fake/Real ratio.
    """
    labels = [item['label'] for item in dataset.video_groups]
    
    # Calculate weights: Weight for class i = 1.0 / count_i
    class_counts = pd.Series(labels).value_counts().to_dict()
    weights = [1.0 / class_counts[label] for label in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader
