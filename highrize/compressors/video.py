"""
Video compressor — extracts keyframes for vision AI instead of sending raw video.

Most vision LLMs don't accept raw video. You sample frames, then send those
as images. This module:
  1. Samples N frames (uniform or scene-change based)
  2. Optionally detects scene changes using frame diff
  3. Compresses each frame via ImageCompressor
  4. Returns list of base64 frames ready for any vision API
"""

import io
import base64
from typing import List, Optional, Tuple, Union
from ..models import CompressionResult, Modality
from .image import ImageCompressor, _image_tokens_estimate


def _cv2_available():
    try:
        import cv2
        return True
    except ImportError:
        return False


class VideoCompressor:
    """
    Extracts and compresses video frames for LLM vision input.

    Args:
        max_frames: Maximum number of frames to extract
        frame_size: Resize each frame to this (w, h) before encoding
        quality: JPEG quality per frame
        scene_change: If True, use frame diff to pick scene-change frames
                      instead of uniform sampling (requires OpenCV)
        provider: For token estimation
    """

    def __init__(
        self,
        token_counter=None,
        max_frames: int = 10,
        frame_size: Tuple[int, int] = (768, 768),
        quality: int = 70,
        scene_change: bool = False,
        provider: str = "openai",
    ):
        from ..tokens import TokenCounter
        if not _cv2_available():
            raise ImportError(
                "OpenCV is required for video compression. "
                "Install it with: pip install opencv-python-headless"
            )
        self.max_frames = max_frames
        self.frame_size = frame_size
        self.quality = quality
        self.scene_change = scene_change
        self.provider = provider
        self.token_counter = token_counter or TokenCounter(provider=provider)
        self._img_compressor = ImageCompressor(
            token_counter=self.token_counter,
            max_size=frame_size,
            quality=quality,
            provider=provider,
        )

    def compress(self, video_path: str) -> CompressionResult:
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Original token estimate: all frames (highly conservative)
        original_tokens = total_frames * _image_tokens_estimate(width, height, self.provider)

        # Pick frame indices
        if self.scene_change:
            indices = self._scene_change_indices(cap, total_frames)
        else:
            step = max(1, total_frames // self.max_frames)
            indices = [i for i in range(0, total_frames, step)][: self.max_frames]

        compressed_frames = []
        compressed_tokens = 0

        # Optimization: Sort indices to read as sequentially as possible
        indices.sort()
        
        last_idx = -1
        for idx in indices:
            # Only jump if not at the next frame
            if idx != last_idx + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            
            ret, frame = cap.read()
            last_idx = idx
            
            if not ret:
                continue
            # cv2 gives BGR, convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL
            from PIL import Image
            pil_img = Image.fromarray(frame_rgb)
            result = self._img_compressor.compress(pil_img)
            compressed_frames.append(result.compressed)
            compressed_tokens += result.compressed_tokens

        cap.release()

        return CompressionResult(
            original=video_path,
            compressed=compressed_frames,  # list of b64 image strings
            modality=Modality.VIDEO,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            metadata={
                "total_frames": total_frames,
                "extracted_frames": len(compressed_frames),
                "fps": fps,
                "original_resolution": (width, height),
                "frame_size": self.frame_size,
                "method": "scene_change" if self.scene_change else "uniform",
            },
        )

    def _scene_change_indices(self, cap, total_frames: int) -> List[int]:
        """Return indices where a significant scene change occurs."""
        import cv2
        import numpy as np

        indices = [0]
        prev_gray = None
        step = max(1, total_frames // (self.max_frames * 5))  # sample 5x, pick top N

        diffs = []
        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float)))
                diffs.append((diff, i))
            prev_gray = gray

        # Pick top N by diff score
        diffs.sort(reverse=True)
        top = sorted([i for _, i in diffs[: self.max_frames - 1]])
        return [0] + top
