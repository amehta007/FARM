"""Generate a simple synthetic video for testing."""

import cv2
import numpy as np
from pathlib import Path


def generate_sample_video(
    output_path: str = "data/raw/sample.mp4",
    duration_seconds: int = 10,
    fps: int = 30,
    width: int = 640,
    height: int = 480
):
    """
    Generate a synthetic video with moving rectangles (simulating workers).
    
    Args:
        output_path: Output video file path
        duration_seconds: Video duration in seconds
        fps: Frames per second
        width: Frame width
        height: Frame height
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    total_frames = duration_seconds * fps
    
    print(f"Generating {total_frames} frames at {fps} FPS...")
    
    # Define two "workers" (moving rectangles)
    workers = [
        {
            "start_pos": (50, 100),
            "end_pos": (550, 100),
            "size": (60, 120),
            "color": (0, 255, 0)
        },
        {
            "start_pos": (100, 300),
            "end_pos": (500, 350),
            "size": (50, 100),
            "color": (255, 0, 0)
        }
    ]
    
    for frame_idx in range(total_frames):
        # Create frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 200  # Light gray background
        
        # Add grid lines (simulate farm rows)
        for i in range(0, height, 60):
            cv2.line(frame, (0, i), (width, i), (180, 180, 180), 1)
        
        # Draw workers
        for worker in workers:
            t = frame_idx / total_frames  # Normalized time [0, 1]
            
            # Linear interpolation for position
            start_x, start_y = worker["start_pos"]
            end_x, end_y = worker["end_pos"]
            
            # Add some oscillation for realism
            x = int(start_x + (end_x - start_x) * t)
            y = int(start_y + (end_y - start_y) * t + 20 * np.sin(t * 4 * np.pi))
            
            w, h = worker["size"]
            color = worker["color"]
            
            # Draw rectangle (worker)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
        
        # Add frame number
        cv2.putText(frame, f"Frame {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        out.write(frame)
        
        if (frame_idx + 1) % 100 == 0:
            print(f"  Generated {frame_idx + 1}/{total_frames} frames...")
    
    out.release()
    print(f"✓ Sample video saved to {output_path}")


if __name__ == "__main__":
    generate_sample_video()

