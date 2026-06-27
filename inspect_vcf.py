import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

def main():
    parser = argparse.ArgumentParser(description="Inspect and preview VCF face crops")
    parser.add_argument("--manifest", type=str, required=True, help="Path to face_cache_manifest.csv")
    parser.add_argument("--output", type=str, required=True, help="Path to save preview_crops.jpg")
    parser.add_argument("--grid-size", type=int, default=5, help="Grid size (NxN)")
    args = parser.parse_args()
    
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: Manifest file not found at {manifest_path}")
        return
        
    df = pd.read_csv(manifest_path)
    if df.empty:
        print("Manifest is empty.")
        return
        
    n_images = args.grid_size * args.grid_size
    sample_df = df.sample(n=min(n_images, len(df)))
    
    base_dir = manifest_path.parent
    
    images = []
    for _, row in sample_df.iterrows():
        img_path = base_dir / row["crop_path"]
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
                
    if not images:
        print("No valid images could be loaded from the manifest.")
        return
        
    target_size = images[0].shape[:2]
    resized_images = [cv2.resize(img, (target_size[1], target_size[0])) for img in images]
    
    grid_rows = []
    idx = 0
    for i in range(args.grid_size):
        row_imgs = []
        for j in range(args.grid_size):
            if idx < len(resized_images):
                row_imgs.append(resized_images[idx])
            else:
                row_imgs.append(np.zeros_like(resized_images[0]))
            idx += 1
        grid_rows.append(np.hstack(row_imgs))
        
    grid_img = np.vstack(grid_rows)
    cv2.imwrite(args.output, grid_img)
    print(f"Successfully saved preview grid with {len(images)} images to {args.output}")

if __name__ == "__main__":
    main()
