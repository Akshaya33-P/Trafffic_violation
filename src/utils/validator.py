import os
from typing import Tuple, Dict, Any, Optional

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILE_SIZE_BYTES = 250 * 1024 * 1024  # 250 MB limit

def validate_media_file(file_path: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validates uploaded media file integrity, size, and format.
    Returns:
        (is_valid, error_message, metadata_dict)
    """
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}", {}

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return False, "File is completely empty (0 bytes). Upload rejected.", {}

    if file_size > MAX_FILE_SIZE_BYTES:
        return False, f"File size ({file_size / (1024*1024):.1f} MB) exceeds maximum allowed size (250 MB).", {}

    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in SUPPORTED_IMAGE_EXTENSIONS:
        try:
            import cv2
            img = cv2.imread(file_path)
            if img is None or img.size == 0:
                return False, "Corrupted image file: unable to decode image pixels.", {}
            h, w = img.shape[:2]
            return True, "", {
                "media_type": "image",
                "width": w,
                "height": h,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "extension": ext
            }
        except Exception as e:
            return False, f"Error reading image: {str(e)}", {}

    elif ext in SUPPORTED_VIDEO_EXTENSIONS:
        try:
            import cv2
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return False, "Corrupted video file: VideoCapture could not open file stream.", {}
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Read first frame to ensure decoder can read streams
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                return False, "Corrupted video: Failed to decode the first frame of the video stream.", {}

            # Fallback for FPS if missing or 0
            if fps <= 0 or fps > 240:
                fps = 25.0

            duration_sec = total_frames / fps if total_frames > 0 else 0.0

            return True, "", {
                "media_type": "video",
                "width": width,
                "height": height,
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "duration_seconds": round(duration_sec, 2),
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "extension": ext
            }
        except Exception as e:
            return False, f"Error validating video stream: {str(e)}", {}
    else:
        allowed = list(SUPPORTED_VIDEO_EXTENSIONS.union(SUPPORTED_IMAGE_EXTENSIONS))
        return False, f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(allowed)}", {}
