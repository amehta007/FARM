"""Download and export YOLOv8n model to ONNX format."""

from pathlib import Path
import sys

from loguru import logger
from ultralytics import YOLO


def download_and_export(output_dir: str = "models/weights") -> Path:
    """
    Download YOLOv8n and export to ONNX format for CPU inference.
    
    Args:
        output_dir: Directory to save the ONNX model
    
    Returns:
        Path to the exported ONNX model
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    onnx_path = output_path / "yolov8n.onnx"
    
    if onnx_path.exists():
        logger.info(f"ONNX model already exists at {onnx_path}")
        return onnx_path
    
    logger.info("Downloading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    
    logger.info("Exporting to ONNX format...")
    # Export to ONNX with dynamic batch size and FP32 precision for CPU
    model.export(
        format="onnx",
        dynamic=True,
        simplify=True,
        opset=12,
        imgsz=640
    )
    
    # Move to target location
    source_onnx = Path("yolov8n.onnx")
    if source_onnx.exists():
        source_onnx.rename(onnx_path)
        logger.success(f"Model exported to {onnx_path}")
    else:
        logger.error("Export failed - ONNX file not found")
        sys.exit(1)
    
    return onnx_path


if __name__ == "__main__":
    logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
    download_and_export()

