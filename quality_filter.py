import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import concurrent.futures

def process_chunk(df_chunk_data):
    """
    df_chunk_data is a tuple (chunk_id, rows_list, base_dir, var_thresh, blur_thresh)
    """
    chunk_id, rows_list, base_dir, var_thresh, blur_thresh = df_chunk_data
    
    # Initialize mediapipe locally for this worker to avoid thread safety issues
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    
    results = []
    
    for idx, crop_path in rows_list:
        img_path = base_dir / crop_path
        if not img_path.exists():
            results.append((idx, False))
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            results.append((idx, False))
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Pixel variance (flat image)
        if np.var(gray) < var_thresh:
            results.append((idx, False))
            continue
            
        # 2. Laplacian variance (blur)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < blur_thresh:
            results.append((idx, False))
            continue
            
        # 3. Mediapipe Face Confidence
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        det_result = detector.process(rgb_img)
        
        if not det_result.detections:
            results.append((idx, False))
            continue
            
        # Passed all filters
        results.append((idx, True))
        
    detector.close()
    return results

def main():
    parser = argparse.ArgumentParser(description="Light Quality Filter with Mediapipe")
    parser.add_argument("--manifest", type=str, default="../Data_preprocessing/output_vcf/face_cache_manifest.csv")
    parser.add_argument("--output-manifest", type=str, default="../Data_preprocessing/output_vcf/face_cache_filtered.csv")
    parser.add_argument("--var-thresh", type=float, default=10.0, help="Pixel variance threshold")
    parser.add_argument("--blur-thresh", type=float, default=15.0, help="Laplacian variance threshold")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers for multiprocessing")
    
    args = parser.parse_args()
    
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return
        
    df = pd.read_csv(manifest_path)
    base_dir = manifest_path.parent
    
    print(f"Loaded {len(df)} crops from manifest.")
    print("Running Quality Filter (Variance, Blur, and Mediapipe Face Confidence)...")
    
    # Prepare chunks
    rows_list = [(row.name, row['crop_path']) for idx, row in df.iterrows()]
    chunk_size = len(rows_list) // args.workers + 1
    
    chunks = []
    for i in range(args.workers):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(rows_list))
        if start < end:
            chunks.append((i, rows_list[start:end], base_dir, args.var_thresh, args.blur_thresh))
            
    valid_indices = set()
    total_processed = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_chunk, chunk): chunk for chunk in chunks}
        
        # We use tqdm to track progress of futures
        # Since we only have `workers` futures, it's not very granular, but it's enough.
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing chunks"):
            res = future.result()
            for idx, is_valid in res:
                if is_valid:
                    valid_indices.add(idx)
                    
    filtered_df = df.loc[list(valid_indices)].copy()
    filtered_df.to_csv(args.output_manifest, index=False)
    
    garbage_count = len(df) - len(filtered_df)
    
    print(f"\nFiltering Complete!")
    print(f"Removed {garbage_count} garbage crops.")
    print(f"Kept {len(filtered_df)} valid crops.")
    print(f"Saved filtered manifest to {args.output_manifest}")

if __name__ == "__main__":
    main()
