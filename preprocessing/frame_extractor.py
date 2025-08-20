import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path


class VideoFrameExtractor:
    
    def __init__(self, video_directory, output_directory, annotation_file):
        self.video_dir = Path(video_directory)
        self.output_dir = Path(output_directory)
        self.annotation_file = annotation_file
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Video directory: {self.video_dir}")
        print(f"Output directory: {self.output_dir}")
        
    def extract_frames_from_dataset(self, target_fps=1, max_frames=None):
        
        if self.annotation_file.endswith('.csv'):
            video_data = pd.read_csv(self.annotation_file)
            if 'video_name' in video_data.columns:
                video_names = video_data['video_name'].tolist()
            elif 'filename' in video_data.columns:
                video_names = video_data['filename'].tolist()
            else:
                raise ValueError("CSV file must contain 'video_name' or 'filename' column")
        else:
            raise ValueError("Only CSV annotation files are supported")
        
        print(f"Processing {len(video_names)} videos...")
        
        successful_extractions = 0
        failed_extractions = []
        
        for idx, video_name in enumerate(video_names):
            print(f"Processing video {idx + 1}/{len(video_names)}: {video_name}")
            
            try:
                success = self.extract_frames_from_video(
                    video_name, target_fps, max_frames
                )
                if success:
                    successful_extractions += 1
                else:
                    failed_extractions.append(video_name)
                    
            except Exception as e:
                print(f"Error processing {video_name}: {str(e)}")
                failed_extractions.append(video_name)
        
        print(f"\nExtraction Summary:")
        print(f"Successful: {successful_extractions}")
        print(f"Failed: {len(failed_extractions)}")
        
        if failed_extractions:
            print(f"Failed videos: {failed_extractions}")
    
    def extract_frames_from_video(self, video_name, target_fps=1, max_frames=None):
        
        video_path = self.video_dir / video_name
        
        if not video_path.exists():
            print(f"Video file not found: {video_path}")
            return False
        
        video_capture = cv2.VideoCapture(str(video_path))
        
        if not video_capture.isOpened():
            print(f"Cannot open video file: {video_path}")
            return False
        
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = video_capture.get(cv2.CAP_PROP_FPS)
        
        if original_fps == 0:
            print(f"Invalid FPS for video: {video_name}")
            video_capture.release()
            return False
        
        print(f"   Video info - Total frames: {total_frames}, FPS: {original_fps:.2f}")
        
        frame_interval = max(1, int(original_fps / target_fps))
        
        video_id = Path(video_name).stem
        video_output_dir = self.output_dir / video_id
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        frame_count = 0
        extracted_count = 0
        last_saved_frame = None
        
        while True:
            ret, frame = video_capture.read()
            
            if not ret:
                break
            
            if frame_count % frame_interval == frame_interval // 2:
                frame_filename = video_output_dir / f"{extracted_count:03d}.png"
                cv2.imwrite(str(frame_filename), frame)
                last_saved_frame = frame
                extracted_count += 1
                
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_count += 1
        
        target_frame_count = max_frames if max_frames else 5
        
        if extracted_count < target_frame_count and last_saved_frame is not None:
            for i in range(extracted_count, target_frame_count):
                frame_filename = video_output_dir / f"{i:03d}.png"
                cv2.imwrite(str(frame_filename), last_saved_frame)
        
        video_capture.release()
        
        print(f"   Extracted {extracted_count} frames (padded to {target_frame_count})")
        
        return True
    
    def validate_extraction(self, video_names=None):
        
        if video_names is None:
            video_dirs = [d for d in self.output_dir.iterdir() if d.is_dir()]
        else:
            video_dirs = [self.output_dir / Path(name).stem for name in video_names]
        
        validation_results = {
            'total_videos': len(video_dirs),
            'valid_videos': 0,
            'invalid_videos': [],
            'frame_counts': {}
        }
        
        for video_dir in video_dirs:
            if not video_dir.exists():
                validation_results['invalid_videos'].append(video_dir.name)
                continue
            
            frame_files = list(video_dir.glob("*.png"))
            frame_count = len(frame_files)
            
            validation_results['frame_counts'][video_dir.name] = frame_count
            
            if frame_count > 0:
                validation_results['valid_videos'] += 1
            else:
                validation_results['invalid_videos'].append(video_dir.name)
        
        return validation_results
    
    def print_validation_results(self, results):
        
        print(f"\nFrame Extraction Validation:")
        print(f"=" * 50)
        print(f"Total videos processed: {results['total_videos']}")
        print(f"Valid extractions: {results['valid_videos']}")
        print(f"Invalid extractions: {len(results['invalid_videos'])}")
        
        if results['invalid_videos']:
            print(f"Invalid videos: {results['invalid_videos']}")
        
        frame_counts = list(results['frame_counts'].values())
        if frame_counts:
            print(f"\nFrame count statistics:")
            print(f"  Mean: {np.mean(frame_counts):.1f}")
            print(f"  Min: {np.min(frame_counts)}")
            print(f"  Max: {np.max(frame_counts)}")
            print(f"  Std: {np.std(frame_counts):.1f}")


def main():
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract frames from video dataset')
    parser.add_argument('--video_dir', type=str, required=True,
                       help='Directory containing video files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save extracted frames')
    parser.add_argument('--annotation_file', type=str, required=True,
                       help='CSV file containing video annotations')
    parser.add_argument('--target_fps', type=float, default=1.0,
                       help='Target frame rate for extraction')
    parser.add_argument('--max_frames', type=int, default=5,
                       help='Maximum frames to extract per video')
    parser.add_argument('--validate', action='store_true',
                       help='Validate extraction after completion')
    
    args = parser.parse_args()
    
    extractor = VideoFrameExtractor(
        video_directory=args.video_dir,
        output_directory=args.output_dir,
        annotation_file=args.annotation_file
    )
    
    print("Starting frame extraction...")
    extractor.extract_frames_from_dataset(
        target_fps=args.target_fps,
        max_frames=args.max_frames
    )
    
    if args.validate:
        print("Validating extraction...")
        results = extractor.validate_extraction()
        extractor.print_validation_results(results)
    
    print("Frame extraction completed")


if __name__ == "__main__":
    main()