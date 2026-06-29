import cv2
import os

def list_videos(raw_video_dir):
    videos = []
    for root, _, files in os.walk(raw_video_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                videos.append(os.path.join(root, file))
    return sorted(videos)

def read_frames_with_timestamps(video_path):
    """Yields (frame, timestamp) for a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0 # fallback
        
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp = frame_idx / fps
        yield frame, timestamp, frame_idx
        frame_idx += 1
        
    cap.release()
