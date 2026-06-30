import argparse
import sys
import os

from src.config import load_config
from src.utils import setup_logger

def run_preprocess(config, logger):
    logger.info("Starting preprocessing...")
    
    from src.video_reader import list_videos, read_frames_with_timestamps
    from src.sampler import sample_candidates
    from src.face_processor import process_faces
    from src.selector import select_model_frames
    from src.sequence_builder import build_sequences
    from src.manifest import write_sequences_and_manifest
    
    videos = list_videos(config.data.raw_video_dir)
    logger.info(f"Found {len(videos)} videos.")
    
    # Pre-assign split for each video group to prevent target leakage (GroupShuffleSplit style)
    import random
    from collections import defaultdict
    
    stem_to_videos = defaultdict(list)
    for v in videos:
        base = os.path.basename(v)
        # Group by the prefix (usually the hash in VCF filenames) to group all variants together
        group_id = base.split('_')[0] if '_' in base else os.path.splitext(base)[0]
        stem_to_videos[group_id].append(v)
        
    unique_groups = list(stem_to_videos.keys())
    random.seed(42)
    random.shuffle(unique_groups)
    
    # 80% train, 10% val, 10% test split
    n_groups = len(unique_groups)
    n_train = int(n_groups * 0.8)
    n_val = int(n_groups * 0.1)
    
    train_groups = set(unique_groups[:n_train])
    val_groups = set(unique_groups[n_train:n_train+n_val])
    
    video_to_split = {}
    for group_id, group_vids in stem_to_videos.items():
        if group_id in train_groups:
            split_name = 'train'
        elif group_id in val_groups:
            split_name = 'val'
        else:
            split_name = 'test'
        for v in group_vids:
            video_to_split[v] = split_name
            
    for video in videos:
        logger.info(f"Processing {video}...")
        try:
            frames_iter = read_frames_with_timestamps(video)
            candidates = sample_candidates(frames_iter, config.pipeline.candidate_fps)
            face_records = process_faces(candidates, config.face, video_id=os.path.basename(video))
            model_frames = select_model_frames(face_records, config.pipeline, config.quality, config.motion)
            sequences = build_sequences(model_frames, config.pipeline)
            
            # Extract relative path to preserve directory structure and avoid collisions
            rel_video_path = os.path.relpath(video, config.data.raw_video_dir)
            
            # Resolve label using the directory parts (matching the VCF dataset convention)
            parts = rel_video_path.lower().split(os.sep)
            real_keywords = {"targets", "target", "real", "original", "authentic"}
            label = 0 if any(any(kw in part for kw in real_keywords) for part in parts) else 1
            
            split = video_to_split.get(video, 'train')
            
            write_sequences_and_manifest(sequences, rel_video_path, label, split, config.data, config.pipeline)
            logger.info(f"Generated {len(sequences)} sequences for {video}.")
        except Exception as e:
            logger.error(f"Error processing {video}: {e}")
            
def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection Dataset Pipeline")
    parser.add_argument("--mode", choices=['preprocess'], default='preprocess', help="Run dataset preprocessing")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    logger = setup_logger("pipeline")
    config = load_config(args.config)
    
    if args.mode == 'preprocess':
        run_preprocess(config, logger)

if __name__ == "__main__":
    main()
