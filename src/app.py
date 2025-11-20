"""Streamlit dashboard for worker monitoring system."""

import glob
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Page config
st.set_page_config(
    page_title="Worker Monitoring Dashboard",
    page_icon="👷",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_latest_run(output_dir: str = "data/outputs"):
    """Load the latest run data."""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        return None
    
    # Find latest summary file
    summary_files = list(output_path.glob("summary_*.csv"))
    if not summary_files:
        return None
    
    latest_summary = max(summary_files, key=lambda p: p.stat().st_mtime)
    run_id = latest_summary.stem.replace("summary_", "")
    
    # Load all data for this run
    data = {
        "run_id": run_id,
        "summary": pd.read_csv(latest_summary),
    }
    
    # Try to load history
    history_file = output_path / f"history_{run_id}.parquet"
    if history_file.exists():
        data["history"] = pd.read_parquet(history_file)
    
    # Try to load heatmap
    heatmap_file = output_path / f"heatmap_{run_id}.png"
    if heatmap_file.exists():
        data["heatmap"] = str(heatmap_file)
    
    # Try to load annotated video
    video_file = output_path / f"annotated_{run_id}.mp4"
    if video_file.exists():
        data["video"] = str(video_file)
    
    return data


def clear_run_outputs(output_dir: str, raw_dir: str = "data/raw") -> dict:
    """
    Remove generated output artifacts and temporary webcam recordings.

    Args:
        output_dir: Directory containing run outputs (metrics, annotated videos, etc.)
        raw_dir: Directory containing raw input videos (only webcam_* files will be removed)

    Returns:
        Dictionary describing the removed files.
    """
    removed_outputs = []
    removed_webcam = []

    output_path = Path(output_dir)
    if output_path.exists():
        for item in output_path.iterdir():
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
                else:
                    shutil.rmtree(item)
                removed_outputs.append(item.name)
            except Exception as exc:  # pragma: no cover - safety net for rare filesystem errors
                removed_outputs.append(f"{item.name} (failed: {exc})")

    raw_path = Path(raw_dir)
    if raw_path.exists():
        for webcam_file in raw_path.glob("webcam_*.mp4"):
            try:
                webcam_file.unlink()
                removed_webcam.append(webcam_file.name)
            except Exception as exc:  # pragma: no cover
                removed_webcam.append(f"{webcam_file.name} (failed: {exc})")

    return {
        "outputs": removed_outputs,
        "webcam": removed_webcam,
    }


def main():
    """Main dashboard application."""
    
    # Header
    st.title("👷 Worker Monitoring Dashboard")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        if "cleanup_result" in st.session_state:
            cleanup_result = st.session_state.pop("cleanup_result")
            outputs_count = len([name for name in cleanup_result["outputs"] if "failed" not in name])
            webcam_count = len([name for name in cleanup_result["webcam"] if "failed" not in name])
            st.success(
                f"Cleared {outputs_count} output file(s) "
                f"and {webcam_count} webcam recording(s)."
            )
        
        output_dir = st.text_input("Output Directory", value="data/outputs")
        
        if st.button("🔄 Refresh Data"):
            st.rerun()
        if st.button("🧹 Clear logs & videos"):
            cleanup_result = clear_run_outputs(output_dir)
            st.session_state["cleanup_result"] = cleanup_result
            st.rerun()
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This dashboard displays worker detection and tracking results.
        
        **Features:**
        - Real-time worker detection
        - Activity metrics (active/idle time)
        - Zone occupancy analysis
        - Heatmap visualization
        """)
    
    # Load data
    data = load_latest_run(output_dir)
    
    if data is None:
        st.warning("⚠️ No run data found. Please process a video first.")
        st.markdown("""
        ### Quick Start
        
        1. Install dependencies:
           ```bash
           pip install -r requirements.txt
           python -m src.models.download_models
           ```
        
        2. Process a video:
           ```bash
           python -m src.main process --video data/raw/sample.mp4 --save-annotated
           ```
        
        3. Refresh this dashboard
        """)
        return
    
    # Display run info
    st.success(f"✅ Loaded run: **{data['run_id']}**")
    
    # Main content
    tabs = st.tabs(["📊 Overview", "📈 Detailed Metrics", "🗺️ Heatmap", "🎥 Video"])
    
    # Tab 1: Overview
    with tabs[0]:
        st.header("Overview")
        
        summary_df = data["summary"]
        
        # Handle backward compatibility: rename old column name if it exists
        if "active_ratio" in summary_df.columns and "Productivity_measure" not in summary_df.columns:
            summary_df = summary_df.rename(columns={"active_ratio": "Productivity_measure"})
        
        # Calculate video duration from history if available
        video_duration = None
        if "history" in data and not data["history"].empty:
            history_df = data["history"]
            max_timestamp = history_df["timestamp"].max()
            min_timestamp = history_df["timestamp"].min()
            # Estimate FPS from frame indices
            max_frame = history_df["frame_idx"].max()
            if max_frame > 0 and max_timestamp > 0:
                estimated_fps = max_frame / max_timestamp
                video_duration = max_timestamp + (1.0 / estimated_fps)  # Add one frame duration
            else:
                video_duration = max_timestamp
        
        # KPIs
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Workers", len(summary_df))
        
        with col2:
            # Use Productivity_measure if available, otherwise calculate from active/presence times
            if "Productivity_measure" in summary_df.columns:
                avg_active = summary_df["Productivity_measure"].mean() * 100 if not summary_df.empty else 0
            elif "active_time_s" in summary_df.columns and "presence_time_s" in summary_df.columns:
                # Calculate productivity measure from active and presence times
                productivity = summary_df["active_time_s"] / summary_df["presence_time_s"]
                avg_active = productivity.mean() * 100 if not summary_df.empty else 0
            else:
                avg_active = 0
            st.metric("Avg Productivity", f"{avg_active:.1f}%")
        
        with col3:
            if video_duration is not None:
                st.metric("Video Duration", f"{video_duration:.1f}s")
            else:
                st.metric("Video Duration", "N/A")
        
        with col4:
            total_worker_seconds = summary_df["presence_time_s"].sum() if not summary_df.empty else 0
            st.metric("Total Worker-Seconds", f"{total_worker_seconds:.1f}s")
        
        with col5:
            total_idle = summary_df["idle_time_s"].sum() if not summary_df.empty else 0
            st.metric("Total Idle Time", f"{total_idle:.1f}s")
        
        # Explanation of metrics
        if video_duration is not None and not summary_df.empty:
            st.info(f"💡 **Note:** Video Duration ({video_duration:.1f}s) is the actual video length. "
                   f"Total Worker-Seconds ({total_worker_seconds:.1f}s) is the sum of all individual worker presence times. "
                   f"If multiple workers are present simultaneously, their times are added together.")
        
        st.markdown("---")
        
        # Worker table
        st.subheader("Worker Summary")
        
        if not summary_df.empty:
            # Format for display
            display_df = summary_df.copy()
            # Handle Productivity_measure column (with backward compatibility)
            if "Productivity_measure" in display_df.columns:
                display_df["Productivity_measure"] = display_df["Productivity_measure"].apply(lambda x: f"{x*100:.1f}%")
            elif "active_ratio" in display_df.columns:
                # Rename and format old column
                display_df = display_df.rename(columns={"active_ratio": "Productivity_measure"})
                display_df["Productivity_measure"] = display_df["Productivity_measure"].apply(lambda x: f"{x*100:.1f}%")
            elif "active_time_s" in display_df.columns and "presence_time_s" in display_df.columns:
                # Calculate if not present
                display_df["Productivity_measure"] = (display_df["active_time_s"] / display_df["presence_time_s"]).apply(lambda x: f"{x*100:.1f}%")
            
            display_df["presence_time_s"] = display_df["presence_time_s"].apply(lambda x: f"{x:.1f}s")
            display_df["active_time_s"] = display_df["active_time_s"].apply(lambda x: f"{x:.1f}s")
            display_df["idle_time_s"] = display_df["idle_time_s"].apply(lambda x: f"{x:.1f}s")
            
            st.dataframe(display_df, use_container_width=True)
            
            # Download button
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Summary CSV",
                data=csv,
                file_name=f"summary_{data['run_id']}.csv",
                mime="text/csv"
            )
        else:
            st.info("No workers detected in this run.")
        
        st.markdown("---")
        
        # Charts
        if not summary_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Active vs Idle Time")
                chart_data = summary_df[["track_id", "active_time_s", "idle_time_s"]].set_index("track_id")
                st.bar_chart(chart_data)
            
            with col2:
                st.subheader("Presence Time Distribution")
                st.bar_chart(summary_df.set_index("track_id")["presence_time_s"])
    
    # Tab 2: Detailed Metrics
    with tabs[1]:
        st.header("Detailed Metrics")
        
        if "history" in data:
            history_df = data["history"]
            
            # Track selector
            track_ids = sorted(history_df["track_id"].unique())
            selected_track = st.selectbox("Select Worker ID", track_ids)
            
            # Filter data
            track_data = history_df[history_df["track_id"] == selected_track]
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_frames = len(track_data)
                st.metric("Total Frames", total_frames)
            
            with col2:
                active_frames = track_data["is_active"].sum()
                st.metric("Active Frames", active_frames)
            
            with col3:
                avg_speed = track_data["speed_px_s"].mean()
                st.metric("Avg Speed", f"{avg_speed:.1f} px/s")
            
            st.markdown("---")
            
            # Timeline
            st.subheader("Activity Timeline")
            timeline_data = track_data[["frame_idx", "speed_px_s"]].set_index("frame_idx")
            st.line_chart(timeline_data)
            
            st.markdown("---")
            
            # Trajectory
            st.subheader("Movement Trajectory")
            trajectory_data = track_data[["center_x", "center_y"]]
            st.scatter_chart(trajectory_data.rename(columns={"center_x": "x", "center_y": "y"}))
            
            # Download
            csv = track_data.to_csv(index=False)
            st.download_button(
                label=f"📥 Download Track {selected_track} Data",
                data=csv,
                file_name=f"track_{selected_track}_{data['run_id']}.csv",
                mime="text/csv"
            )
        else:
            st.info("Per-frame history not available. Enable save_per_frame in config.")
    
    # Tab 3: Heatmap
    with tabs[2]:
        st.header("Occupancy Heatmap")
        
        if "heatmap" in data:
            st.subheader("Time-based Occupancy")
            st.image(data["heatmap"], use_column_width=True, caption="Heatmap showing where workers spend most time")
            
            st.markdown("""
            **How to read this heatmap:**
            - Warmer colors (red/yellow) indicate high occupancy
            - Cooler colors (blue/purple) indicate low occupancy
            - White/black areas indicate no worker presence
            """)
        else:
            st.info("Heatmap not available for this run.")
    
    # Tab 4: Video
    with tabs[3]:
        st.header("Annotated Video")
        
        if "video" in data:
            st.subheader("Detection Results")
            
            video_file = open(data["video"], "rb")
            video_bytes = video_file.read()
            
            st.video(video_bytes)
            
            st.markdown(f"**Video file:** `{data['video']}`")
        else:
            st.info("Annotated video not available. Enable --save-annotated when processing.")


if __name__ == "__main__":
    main()

