import os
import json
import hashlib
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from dataclasses import asdict

from preprocess_vcf import extract_faces_task, CropMetadata

def main():
    root_dir = Path("/Users/admin/Documents/HK-4/TDTT/DA_CK/Code/Data_preprocessing/vcf")
    output_dir = Path("/Users/admin/Documents/HK-4/TDTT/DA_CK/Code/Data_preprocessing/output_vcf")
    splits_csv = output_dir / "vcf_manifest_splits.csv"
    
    if not splits_csv.exists():
        print("vcf_manifest_splits.csv not found!")
        return
        
    df = pd.read_csv(splits_csv)
    print(f"Total videos in splits manifest: {len(df)}")
    
    face_cache_dir = output_dir / "face_cache"
    
    print("Scanning and indexing face_cache directory...")
    existing_crops_by_prefix = {}
    for root, _, files in os.walk(face_cache_dir):
        for f in files:
            if f.endswith(".jpg") and "_f" in f:
                prefix = f.rsplit("_f", 1)[0]
                existing_crops_by_prefix.setdefault(prefix, []).append(f)
                
    print("Indexing complete. Identifying missing videos and building existing metadata...")
    
    all_crop_metadata = []
    extraction_tasks = []
    
    max_frames_per_video = 12
    face_size = 224
    jpeg_quality = 90
    workers = 4 
    
    for idx, row in df.iterrows():
        rel_path = row["relative_path"]
        stem = row["stem"]
        split = row["split"]
        label_name = row["label_name"]
        
        path_hash = hashlib.md5(rel_path.encode("utf-8")).hexdigest()[:8]
        prefix = f"{stem}_{path_hash}"
        
        existing_files = existing_crops_by_prefix.get(prefix, [])
        
        if existing_files:
            for filename in existing_files:
                frame_idx_str = filename.split("_f")[-1].replace(".jpg", "")
                try:
                    frame_idx = int(frame_idx_str)
                except ValueError:
                    frame_idx = 0
                    
                crop_path = f"face_cache/{split}/{label_name}/{filename}"
                
                meta = CropMetadata(
                    crop_path=crop_path,
                    source_video=row["filename"],
                    relative_path=rel_path,
                    frame_idx=frame_idx,
                    split=split,
                    label=row["label"],
                    label_name=label_name,
                    compression=row["compression"],
                    method=row["method"],
                    resolution=row["resolution"],
                    background=row["background"],
                    group_id=row["group_id"]
                )
                all_crop_metadata.append(asdict(meta))
        else:
            if not row["readable"]:
                continue
                
            extraction_tasks.append({
                "video_path": root_dir / rel_path,
                "rel_path": rel_path,
                "output_dir": output_dir,
                "split": split,
                "label_name": label_name,
                "metadata": row.to_dict(),
                "max_frames": max_frames_per_video,
                "face_size": face_size,
                "jpeg_quality": jpeg_quality
            })
            
    print(f"Found {len(all_crop_metadata)} existing crops.")
    print(f"Found {len(extraction_tasks)} missing videos to process.")
    
    if extraction_tasks:
        print("Starting face extraction for missing videos...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(extract_faces_task, task): task for task in extraction_tasks}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Extracting missing faces"):
                try:
                    res = future.result()
                    if res:
                        all_crop_metadata.extend(res)
                except Exception as e:
                    print(f"Error extracting faces: {e}")
                    
    print(f"Total crops after processing: {len(all_crop_metadata)}")
    
    crops_df = pd.DataFrame(all_crop_metadata)
    manifest_path = output_dir / "face_cache_manifest.csv"
    
    if not crops_df.empty:
        crops_df.to_csv(manifest_path, index=False)
        print(f"Saved manifest to {manifest_path}")
    else:
        print("No crops found or extracted!")
        
    summary_path = output_dir / "dataset_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
            
        summary["face_crops_extracted"] = len(crops_df)
        
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        print(f"Updated {summary_path} with face_crops_extracted = {len(crops_df)}")
        
if __name__ == "__main__":
    main()
