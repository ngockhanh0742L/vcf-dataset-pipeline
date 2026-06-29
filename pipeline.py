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
    
    for video in videos:
        logger.info(f"Processing {video}...")
        try:
            frames_iter = read_frames_with_timestamps(video)
            candidates = sample_candidates(frames_iter, config.pipeline.candidate_fps)
            face_records = process_faces(candidates, config.face, video_id=os.path.basename(video))
            model_frames = select_model_frames(face_records, config.pipeline, config.quality, config.motion)
            sequences = build_sequences(model_frames, config.pipeline)
            
            # Simple assumption: label from path or fixed for now
            # Real implementation should extract label and split
            label = 0 if 'real' in video.lower() else 1
            split = 'train' # simple default
            
            write_sequences_and_manifest(sequences, os.path.basename(video), label, split, config.data, config.pipeline)
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
