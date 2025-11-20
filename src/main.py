"""Main CLI application for worker detection and tracking."""

import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import typer
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.analytics.heatmap import OccupancyHeatmap
from src.analytics.metrics import MetricsTracker
from src.analytics.zones import ZoneManager
from src.detection.detector import ONNXDetector
from src.io.sink import MetricsSink, VideoWriter
from src.io.video_reader import VideoReader
from src.tracking.bytetrack import ByteTracker
from src.utils.geometry import bbox_center
from src.utils.settings import Config
from src.utils.timing import FPSCounter
from src.vis.draw import blur_bbox, draw_bbox, draw_info_panel, draw_zones

app = typer.Typer(help="Worker Detection and Monitoring System")
console = Console()


@app.command()
def process(
    video: str = typer.Option(None, "--video", "-v", help="Video file path"),
    webcam: int = typer.Option(None, "--webcam", "-w", help="Webcam device index"),
    config: str = typer.Option("src/configs/default.yaml", "--config", "-c", help="Config file path"),
    save_annotated: bool = typer.Option(False, "--save-annotated", "-s", help="Save annotated video"),
    display: bool = typer.Option(True, "--display/--no-display", help="Display video in window"),
):
    """
    Process video or webcam stream for worker detection and tracking.
    """
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}", level="INFO")
    
    console.print("[bold cyan]Worker Detection & Monitoring System[/bold cyan]")
    console.print("=" * 60)
    
    # Load configuration
    cfg = Config.from_yaml(config)
    logger.info(f"Loaded config from {config}")
    
    # Determine video source
    if webcam is not None:
        source = f"webcam:{webcam}"
    elif video is not None:
        source = video
    else:
        source = cfg.video.source
    
    logger.info(f"Video source: {source}")
    
    # Initialize components
    try:
        console.print("[yellow]Initializing detector...[/yellow]")
        detector = ONNXDetector(
            model_path=cfg.model.path,
            conf_threshold=cfg.model.conf_threshold,
            iou_threshold=cfg.model.iou_threshold,
            input_size=tuple(cfg.model.input_size)
        )
        
        console.print("[yellow]Initializing tracker...[/yellow]")
        tracker = ByteTracker(
            track_thresh=cfg.tracking.track_thresh,
            track_buffer=cfg.tracking.track_buffer,
            match_thresh=cfg.tracking.match_thresh,
            min_box_area=cfg.tracking.min_box_area,
            min_track_hits=cfg.tracking.min_track_hits,
            duplicate_iou_thresh=cfg.tracking.duplicate_iou_thresh
        )
        
        console.print("[yellow]Opening video source...[/yellow]")
        video_reader = VideoReader(source, fps_override=cfg.video.fps_override)
        
        fps = video_reader.get_fps()
        frame_shape = video_reader.get_frame_shape()
        
        # Initialize analytics
        metrics_tracker = MetricsTracker(
            idle_speed_threshold=cfg.metrics.idle_speed_px_s,
            fps=fps,
            smoothing_window=cfg.metrics.smoothing_window
        )
        
        zone_manager = ZoneManager([z.model_dump() for z in cfg.zones])
        
        heatmap = OccupancyHeatmap(
            frame_shape=frame_shape,
            grid_size=tuple(cfg.metrics.heatmap_grid)
        )
        
        # Initialize output
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_sink = MetricsSink(cfg.output.dir, run_id=run_id)
        
        video_writer = None
        if save_annotated or cfg.output.save_annotated:
            output_video_path = Path(cfg.output.dir) / f"annotated_{run_id}.mp4"
            video_writer = VideoWriter(
                str(output_video_path),
                (video_reader.width, video_reader.height),
                fps
            )
        
        # Save configuration
        metrics_sink.save_config(cfg.model_dump())
        
        console.print("[bold green]Processing started![/bold green]")
        
        # Processing loop
        fps_counter = FPSCounter()
        frame_count = 0
        
        # Use simple progress without Unicode spinner for Windows compatibility
        try:
            with Progress(TextColumn("[progress.description]{task.description}")) as progress:
                task = progress.add_task("Processing frames...", total=None)
                
                while True:
                    ret, frame = video_reader.read()
                    if not ret:
                        break
                    
                    frame_idx = video_reader.get_current_frame_idx()
                    timestamp = frame_idx / fps
                    
                    # Detection (skip frames for speed)
                    if frame_idx % cfg.video.skip_frames == 0:
                        detections = detector.detect(frame)
                    else:
                        detections = np.empty((0, 5))
                    
                    # Tracking
                    tracks = tracker.update(detections)
                    
                    # Process each track
                    for track in tracks:
                        track_id = int(track[4])
                        bbox = track[:4]
                        score = track[5]
                        
                        # Update metrics
                        metrics_tracker.update(track_id, bbox, timestamp, frame_idx)
                        
                        # Update zones
                        center = bbox_center(bbox)
                        zone_manager.update(track_id, center, 1.0 / fps)
                        
                        # Update heatmap
                        heatmap.update(center, 1.0 / fps)
                    
                    # Visualization
                    vis_frame = frame.copy()
                    
                    # Draw zones
                    vis_frame = draw_zones(vis_frame, zone_manager.zones, alpha=0.2)
                    
                    # Draw tracks
                    for track in tracks:
                        track_id = int(track[4])
                        bbox = track[:4]
                        score = track[5]
                        
                        # Get active/idle status for visualization
                        is_active = metrics_tracker.is_track_active(track_id)
                        
                        # Blur faces if enabled
                        if cfg.output.blur_faces:
                            vis_frame = blur_bbox(vis_frame, bbox, kernel_size=31)
                        
                        # Draw bounding box with active/idle color coding
                        vis_frame = draw_bbox(vis_frame, bbox, track_id, score, is_active=is_active)
                    
                    # Draw info panel
                    current_fps = fps_counter.tick()
                    info = {
                        "Frame": f"{frame_idx}",
                        "FPS": f"{current_fps:.1f}" if current_fps else "N/A",
                        "Workers": len(tracks),
                        "Tracks": len(metrics_tracker.tracks)
                    }
                    vis_frame = draw_info_panel(vis_frame, info, position="top-right")
                    
                    # Display
                    if display:
                        cv2.imshow("Worker Detection", vis_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logger.info("User requested quit")
                            break
                    
                    # Write annotated video
                    if video_writer is not None:
                        video_writer.write(vis_frame)
                    
                    frame_count += 1
                    
                    if frame_count % 100 == 0:
                        logger.info(f"Processed {frame_count} frames...")
                        progress.update(task, description=f"Processed {frame_count} frames...")
        except UnicodeEncodeError:
            # Fallback for Windows console encoding issues
            logger.info("Processing frames (progress display disabled for Windows compatibility)...")
            while True:
                ret, frame = video_reader.read()
                if not ret:
                    break
                
                frame_idx = video_reader.get_current_frame_idx()
                timestamp = frame_idx / fps
                
                # Detection (skip frames for speed)
                if frame_idx % cfg.video.skip_frames == 0:
                    detections = detector.detect(frame)
                else:
                    detections = np.empty((0, 5))
                
                # Tracking
                tracks = tracker.update(detections)
                
                # Process each track
                for track in tracks:
                    track_id = int(track[4])
                    bbox = track[:4]
                    score = track[5]
                    
                    # Update metrics
                    metrics_tracker.update(track_id, bbox, timestamp, frame_idx)
                    
                    # Update zones
                    center = bbox_center(bbox)
                    zone_manager.update(track_id, center, 1.0 / fps)
                    
                    # Update heatmap
                    heatmap.update(center, 1.0 / fps)
                
                # Visualization
                vis_frame = frame.copy()
                
                # Draw zones
                vis_frame = draw_zones(vis_frame, zone_manager.zones, alpha=0.2)
                
                # Draw tracks
                for track in tracks:
                    track_id = int(track[4])
                    bbox = track[:4]
                    score = track[5]
                    
                    # Get active/idle status for visualization
                    is_active = metrics_tracker.is_track_active(track_id)
                    
                    # Blur faces if enabled
                    if cfg.output.blur_faces:
                        vis_frame = blur_bbox(vis_frame, bbox, kernel_size=31)
                    
                    # Draw bounding box with active/idle color coding
                    vis_frame = draw_bbox(vis_frame, bbox, track_id, score, is_active=is_active)
                
                # Draw info panel
                current_fps = fps_counter.tick()
                info = {
                    "Frame": f"{frame_idx}",
                    "FPS": f"{current_fps:.1f}" if current_fps else "N/A",
                    "Workers": len(tracks),
                    "Tracks": len(metrics_tracker.tracks)
                }
                vis_frame = draw_info_panel(vis_frame, info, position="top-right")
                
                # Display
                if display:
                    cv2.imshow("Worker Detection", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User requested quit")
                        break
                
                # Write annotated video
                if video_writer is not None:
                    video_writer.write(vis_frame)
                
                frame_count += 1
                
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames...")
        
        # Cleanup
        video_reader.release()
        if video_writer is not None:
            video_writer.release()
        if display:
            cv2.destroyAllWindows()
        
        console.print(f"[bold green]Processing complete! Processed {frame_count} frames[/bold green]")
        
        # Save metrics
        console.print("[yellow]Saving metrics...[/yellow]")
        
        summary_df = metrics_tracker.get_all_summaries()
        if not summary_df.empty:
            # Add zone times
            for track_id in summary_df["track_id"]:
                zone_times = zone_manager.get_all_zone_times(track_id)
                for zone_name, time_val in zone_times.items():
                    summary_df.loc[summary_df["track_id"] == track_id, f"zone_{zone_name}_s"] = time_val
            
            metrics_sink.save_summary(summary_df)
        
        if cfg.output.save_per_frame:
            history_df = metrics_tracker.get_all_histories()
            if not history_df.empty:
                metrics_sink.save_history(history_df)
        
        # Save heatmap
        heatmap_image = heatmap.get_heatmap_image()
        metrics_sink.save_heatmap(heatmap_image)
        
        # Print summary
        console.print("\n[bold cyan]Summary Statistics:[/bold cyan]")
        console.print(f"Total tracks detected: {len(summary_df)}")
        console.print(f"Total frames processed: {frame_count}")
        
        if not summary_df.empty:
            console.print(f"Average productivity measure: {summary_df['Productivity_measure'].mean():.2%}")
            console.print(f"Total worker-seconds: {summary_df['presence_time_s'].sum():.1f}s")
        
        console.print(f"\n[bold green]Results saved to: {cfg.output.dir}/[/bold green]")
        console.print(f"Run ID: {run_id}")
        
    except Exception as e:
        logger.exception("Error during processing")
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def demo():
    """
    Run a quick demo with sample data.
    """
    console.print("[bold cyan]Running demo...[/bold cyan]")
    console.print("This will process the sample video with default settings.\n")
    
    process(
        video="data/raw/sample.mp4",
        config="src/configs/default.yaml",
        save_annotated=True,
        display=False
    )


if __name__ == "__main__":
    app()

