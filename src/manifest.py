import os
import json
import pandas as pd
import uuid
import numpy as np

def write_sequences_and_manifest(sequences, video_id, label, split, config_data, config_pipeline):
    manifest_path = config_data.manifest_path
    seq_dir = config_data.sequence_dir
    os.makedirs(seq_dir, exist_ok=True)
    
    rows = []
    
    for seq_idx, seq in enumerate(sequences):
        seq_id = str(uuid.uuid4())
        seq_folder = os.path.join(seq_dir, seq_id)
        os.makedirs(seq_folder, exist_ok=True)
        
        frame_paths = []
        import cv2
        for i, frame_record in enumerate(seq['frames']):
            frame_name = f"frame_{i:02d}.jpg"
            frame_path = os.path.join(seq_folder, frame_name)
            
            # Save the face crop (converting RGB to BGR for cv2.imwrite)
            if frame_record.face_image is not None:
                bgr_img = cv2.cvtColor(frame_record.face_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(frame_path, bgr_img)
            else:
                # Create a black dummy frame if missing (should not happen with repair policy)
                bgr_img = np.zeros((300, 300, 3), dtype=np.uint8)
                cv2.imwrite(frame_path, bgr_img)
                
            frame_paths.append(os.path.join(seq_id, frame_name))
            
        row = {
            'sequence_id': seq_id,
            'video_id': video_id,
            'label': label,
            'split': split,
            'frame_paths': json.dumps(frame_paths),
            'start_time': seq['frames'][0].timestamp,
            'end_time': seq['frames'][-1].timestamp,
            'candidate_fps': config_pipeline.candidate_fps,
            'model_fps': config_pipeline.model_fps,
            'seq_len': config_pipeline.seq_len,
            'avg_quality': seq['mean_quality'],
            'avg_motion': seq['mean_motion'],
            'duplicate_ratio': seq['duplicate_ratio'],
            'missing_face_ratio': seq['missing_or_repaired_ratio'],
            'repair_ratio': seq['missing_or_repaired_ratio'], # Assuming they are same in this policy
            'window_score': seq['window_score'],
            'pipeline_version': 'b3_300_v1'
        }
        rows.append(row)
        
    if not rows:
        return
        
    df = pd.DataFrame(rows)
    
    if os.path.exists(manifest_path):
        df.to_csv(manifest_path, mode='a', header=False, index=False)
    else:
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        df.to_csv(manifest_path, index=False)
