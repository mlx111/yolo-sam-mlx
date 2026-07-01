"""Remote perception client — runs on the real robot.

One-file convenience wrapper that captures colour + depth from an Intel
RealSense camera, sends them to the remote perception server (running on
the desktop / dev machine), and returns the detected object's 3-D position.

Usage (CLI)
-----------
    # Step 0: make sure the server is running on your dev machine:
    #   http://192.168.1.100:8088 (for example)

    # Step 1 (on the robot): detect an object
    python remote_perception_client.py \\
        --server http://192.168.1.100:8088 \\
        --target-class "red box"

    # Step 1 (with a local image pair for testing):
    python remote_perception_client.py \\
        --server http://127.0.0.1:8088 \\
        --target-class "red box" \\
        --rgb  /path/to/color.png \\
        --depth /path/to/depth.png

Usage (library)
---------------
    from remote_perception_client import RemotePerceptionClient, capture_realsense

    client = RemotePerceptionClient("http://192.168.1.100:8088")

    # --- Option A: capture live from RealSense ---
    rgb, depth = capture_realsense()
    result = client.detect_object(rgb, depth, "red box")
    print(result.position_m)      # [0.52, -0.06, 0.78]

    # --- Option B: use your own images ---
    result = client.detect_object(rgb_img, depth_img, "apple")
    if result.success:
        print(f"Found at {result.position_m} m")

Dependencies (on the robot)
---------------------------
    The robot needs these Python packages:

        pip install requests opencv-python numpy pyrealsense2

    If using the --rgb/--depth file flags, ``pyrealsense2`` is optional.
"""

from __future__ import annotations

import argparse
import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

logger = logging.getLogger("remote_perception_client")


# ---------------------------------------------------------------------------
# Data class for the detection result
# ---------------------------------------------------------------------------


@dataclass
class DetectionResult:
    """Structured result returned by :meth:`RemotePerceptionClient.detect_object`.

    Attributes
    ----------
    success : bool
        Whether the server successfully detected the object.
    target_class : str
        The class name that was searched for.
    coordinate_system : str
        Which coordinate frame the position is expressed in.
    position_m : list[float] | None
        Center of the object in **metres**, e.g. ``[0.52, -0.06, 0.78]``.
        ``None`` if detection failed.
    position_raw_mm : list[float] | None
        Same position in **millimetres**.
    bbox_xyxy : list[float] | None
        Bounding box in pixel space ``[x1, y1, x2, y2]``.
    mask_pixel_count : int
        Number of non-zero pixels in the segmentation mask.
    valid_depth_count : int
        Number of mask pixels that have a valid (>0) depth value.
    point_count : int
        Number of 3-D points after filtering.
    candidate_score : float
        Detection confidence from GroundingDINO (≈ 0–1).
    candidate_label : str
        The text prompt that produced the best detection.
    error : str | None
        Error message if ``success`` is ``False``.
    raw : dict
        The full JSON response from the server (for debugging).
    """

    success: bool = False
    target_class: str = ""
    coordinate_system: str = "world"
    position_m: list[float] | None = None
    position_raw_mm: list[float] | None = None
    bbox_xyxy: list[float] | None = None
    mask_pixel_count: int = 0
    valid_depth_count: int = 0
    point_count: int = 0
    candidate_score: float = 0.0
    candidate_label: str = ""
    error: str | None = None
    raw: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class RemotePerceptionClient:
    """HTTP client for the remote GroundedSAM2 perception server.

    Parameters
    ----------
    server_url : str
        Base URL of the server, e.g. ``"http://192.168.1.100:8088"``.
    timeout_seconds : float
        Request timeout (default 60 s — models can take 10–30 s).
    """

    def __init__(
        self,
        server_url: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Call ``GET /health`` and return the server's status dict."""
        resp = requests.get(
            f"{self.server_url}/health",
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def detect_object(
        self,
        rgb_img: np.ndarray,
        depth_img: np.ndarray,
        target_class: str,
        coordinate_system: str = "world",
        rgb_format: str = "jpg",
        jpeg_quality: int = 90,
    ) -> DetectionResult:
        """Detect *target_class* in the provided images and return its 3-D
        position.

        Parameters
        ----------
        rgb_img : np.ndarray
            BGR colour image, shape ``(480, 640, 3)``, dtype ``uint8``.
        depth_img : np.ndarray
            Depth map, shape ``(480, 640)``, dtype ``uint16``, unit **mm**,
            with 0 for invalid pixels.
        target_class : str
            Object to detect, e.g. ``"red box"``, ``"apple"``.
        coordinate_system : str
            One of ``"camera"``, ``"base"``, ``"world"`` (default ``"world"``).
        rgb_format : str
            ``"jpg"`` (smaller upload, slight loss) or ``"png"`` (lossless).
        jpeg_quality : int
            JPEG quality 0–100 (only used when ``rgb_format="jpg"``).

        Returns
        -------
        DetectionResult
        """
        # ---- encode images to in-memory bytes ----------------------------
        rgb_bytes = self._encode_rgb(rgb_img, fmt=rgb_format, quality=jpeg_quality)
        depth_bytes = self._encode_depth(depth_img)
        if rgb_bytes is None or depth_bytes is None:
            return DetectionResult(
                success=False,
                target_class=target_class,
                error="Failed to encode image",
            )

        # ---- build multipart request -------------------------------------
        rgb_ext = ".jpg" if rgb_format == "jpg" else ".png"
        files = {
            "rgb": (f"rgb{rgb_ext}", rgb_bytes, f"image/{rgb_format}"),
            "depth": ("depth.png", depth_bytes, "image/png"),
        }
        data = {
            "target_class": target_class,
            "coordinate_system": coordinate_system,
        }

        # ---- send --------------------------------------------------------
        try:
            resp = requests.post(
                f"{self.server_url}/detect_pose",
                files=files,
                data=data,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            return DetectionResult(
                success=False,
                target_class=target_class,
                error=f"Connection failed: {exc}",
            )
        except requests.exceptions.Timeout:
            return DetectionResult(
                success=False,
                target_class=target_class,
                error=f"Request timed out ({self.timeout} s)",
            )
        except requests.exceptions.RequestException as exc:
            return DetectionResult(
                success=False,
                target_class=target_class,
                error=str(exc),
            )

        raw: dict = resp.json()
        if not raw.get("success"):
            return DetectionResult(
                success=False,
                target_class=target_class,
                error=raw.get("error", "unknown error"),
                raw=raw,
            )

        return DetectionResult(
            success=True,
            target_class=raw.get("target_class", target_class),
            coordinate_system=raw.get("coordinate_system", coordinate_system),
            position_m=raw.get("position_m"),
            position_raw_mm=raw.get("position_raw_mm"),
            bbox_xyxy=raw.get("bbox_xyxy"),
            mask_pixel_count=raw.get("mask_pixel_count", 0),
            valid_depth_count=raw.get("valid_depth_count", 0),
            point_count=raw.get("point_count", 0),
            candidate_score=raw.get("candidate", {}).get("score", 0.0),
            candidate_label=raw.get("candidate", {}).get("label", ""),
            raw=raw,
        )

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_rgb(
        rgb_img: np.ndarray,
        fmt: str = "jpg",
        quality: int = 90,
    ) -> bytes | None:
        if fmt == "jpg":
            ok, buf = cv2.imencode(".jpg", rgb_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            ok, buf = cv2.imencode(".png", rgb_img)
        return buf.tobytes() if ok else None

    @staticmethod
    def _encode_depth(depth_img: np.ndarray) -> bytes | None:
        if depth_img.dtype != np.uint16:
            depth_img = depth_img.astype(np.uint16)
        ok, buf = cv2.imencode(".png", depth_img)
        return buf.tobytes() if ok else None


# ---------------------------------------------------------------------------
# Live capture helper  (for use on the robot with a RealSense D435 / D415)
# ---------------------------------------------------------------------------


def capture_realsense(
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    timeout_sec: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Capture one aligned colour + depth frame from an Intel RealSense camera.

    Requires ``pyrealsense2`` installed on the robot.

    Parameters
    ----------
    width, height : int
        Resolution (must match server expectations — default 640×480).
    fps : int
        Frame rate.
    timeout_sec : float
        How long to wait for a valid depth frame.

    Returns
    -------
    rgb : np.ndarray
        BGR image, shape ``(height, width, 3)``, dtype ``uint8``.
    depth : np.ndarray
        Depth map, shape ``(height, width)``, dtype ``uint16``, unit **mm**.
        0 = invalid.

    Raises
    ------
    ImportError
        If ``pyrealsense2`` is not installed.
    RuntimeError
        If no valid frame is received within *timeout_sec*.
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        raise ImportError(
            "capture_realsense() requires pyrealsense2.\n"
            "Install it on the robot with:\n"
            "    pip install pyrealsense2"
        )

    # --- configure pipeline ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    try:
        profile = pipeline.start(config)
        # Enable alignment of depth to colour.
        align_to = rs.stream.color
        align = rs.align(align_to)

        # Wait for a valid frame pair.
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if color_frame and depth_frame:
                rgb = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data())
                return rgb, depth

        raise RuntimeError(
            f"No valid frame pair received within {timeout_sec} s"
        )

    finally:
        pipeline.stop()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Remote perception client — detect object via HTTP",
    )
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:8088",
        help="Remote perception server URL (default: http://127.0.0.1:8088)",
    )
    parser.add_argument(
        "--target-class",
        default="red box",
        help="Object class to detect (default: 'red box')",
    )
    parser.add_argument(
        "--coordinate-system",
        default="world",
        choices=["camera", "base", "world"],
        help="Target coordinate frame (default: world)",
    )
    parser.add_argument(
        "--rgb",
        help="Path to a colour PNG/JPG file (omit to use live RealSense)",
    )
    parser.add_argument(
        "--depth",
        help="Path to a depth uint16 PNG file (omit to use live RealSense)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP request timeout in seconds (default: 60)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    client = RemotePerceptionClient(args.server, timeout_seconds=args.timeout)

    # Health check first.
    try:
        health = client.health()
        logger.info("Server: %s  |  model_loaded=%s", args.server, health.get("model_loaded"))
    except Exception as exc:
        logger.error("Cannot reach server at %s: %s", args.server, exc)
        return

    # Acquire images.
    if args.rgb and args.depth:
        rgb = cv2.imread(args.rgb)
        depth = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            logger.error("Failed to load images from %s / %s", args.rgb, args.depth)
            return
        if depth.ndim == 3 and depth.shape[2] == 1:
            depth = depth.squeeze(axis=2)
        if depth.dtype != np.uint16:
            # Some PNG readers may return uint8; convert if needed.
            depth = depth.astype(np.uint16)
        logger.info("Loaded images: rgb=%s, depth=%s dtype=%s", rgb.shape, depth.shape, depth.dtype)
    else:
        logger.info("Capturing from RealSense camera …")
        try:
            rgb, depth = capture_realsense()
            logger.info("Captured: rgb=%s, depth=%s", rgb.shape, depth.shape)
        except ImportError:
            logger.error(
                "pyrealsense2 not available. Either install it or "
                "provide --rgb / --depth file paths."
            )
            return
        except RuntimeError as exc:
            logger.error("Camera capture failed: %s", exc)
            return

    # Detect.
    logger.info("Detecting '%s' …", args.target_class)
    result = client.detect_object(
        rgb, depth,
        target_class=args.target_class,
        coordinate_system=args.coordinate_system,
    )

    # Print result.
    if result.success:
        print()
        print("=" * 50)
        print(f"  DETECTED:    {result.target_class}")
        print(f"  Position:    {result.position_m}  (metres)")
        print(f"  Position:    {result.position_raw_mm}  (mm)")
        print(f"  Bbox:        {result.bbox_xyxy}")
        print(f"  Mask pixels: {result.mask_pixel_count}")
        print(f"  Points:      {result.point_count}")
        print(f"  Confidence:  {result.candidate_score:.3f}")
        print(f"  Prompt:      {result.candidate_label}")
        print("=" * 50)
    else:
        print()
        print(f"  Detection FAILED: {result.error}")
        print()


if __name__ == "__main__":
    _main()
