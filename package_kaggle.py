import argparse
import os
import zipfile
from tqdm import tqdm
from pathlib import Path

def zip_directory(dir_path, zip_path):
    dir_path = Path(dir_path)
    zip_path = Path(zip_path)
    
    print(f"Packaging {dir_path} into {zip_path}...")
    
    # Get all files first for the progress bar
    all_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            all_files.append(Path(root) / file)
            
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in tqdm(all_files, desc="Zipping files"):
            arcname = file.relative_to(dir_path)
            zipf.write(file, arcname)
            
    print(f"Successfully created {zip_path} ({zip_path.stat().st_size / (1024*1024):.2f} MB)")

def main():
    parser = argparse.ArgumentParser(description="Package dataset for Kaggle")
    parser.add_argument("--input", type=str, required=True, help="Path to output_vcf directory")
    parser.add_argument("--output", type=str, default="vcf_dataset_kaggle.zip", help="Output zip file path")
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Directory {args.input} does not exist.")
        return
        
    zip_directory(args.input, args.output)

if __name__ == "__main__":
    main()
