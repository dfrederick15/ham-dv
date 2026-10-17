"""Video sources: webcam, video file, image, or the built-in synthetic test scene."""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

from .testvideo import synth_frame


class VideoSource:
    def __init__(self, spec: str, width: int, height: int):
        self.w, self.h = width, height
        self.spec = spec
        self.cap = None
        self.kind = "test"
        self.t0 = time.time()
        self.still = None
        self.file_fps = 25.0
        self.file_pos = 0.0
        if spec in (None, "", "test"):
            return
        if spec.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            from PIL import Image
            self.still = np.asarray(Image.open(spec).convert("RGB").resize((width, height), Image.LANCZOS))
            self.kind = "still"
            return
        import cv2
        self.cv2 = cv2
        if spec.isdigit():
            self.cap = cv2.VideoCapture(int(spec))
            self.kind = "camera"
        else:
            if not os.path.exists(spec):
                raise FileNotFoundError(spec)
            self.cap = cv2.VideoCapture(spec)
            self.kind = "file"
            self.file_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video source {spec}")

    def read(self, t: Optional[float] = None) -> np.ndarray:
        """Return an RGB (h, w, 3) frame. t = media time in seconds (used for test/file sources)."""
        if self.kind == "test":
            return synth_frame(time.time() - self.t0 if t is None else t, self.w, self.h)
        if self.kind == "still":
            return self.still
        cv2 = self.cv2
        if self.kind == "file" and t is not None:
            # advance to the requested media time (loop at the end)
            target = int(t * self.file_fps)
            total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total > 0:
                target %= total
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = self.cap.read()
        if not ok:
            if self.kind == "file":
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if not ok:
                return np.full((self.h, self.w, 3), 128, np.uint8)
        frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self):
        if self.cap is not None:
            self.cap.release()
