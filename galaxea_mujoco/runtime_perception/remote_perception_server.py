"""Remote perception server for Galaxea.

Provides HTTP endpoints for robots to call GroundedSAM2 + point-cloud
position estimation remotely via multipart form upload.

Endpoints
---------
GET  /health      — Health check (model loaded, intrinsics, …)
POST /detect_pose — Detect object -> return 3D position in meters

Protocol conventions (MUST match robot-side camera)
---------------------------------------------------
    depth format:      16-bit PNG
    depth dtype:       uint16
    depth unit:        millimeter  (0 = invalid)
    rgb shape:         640 x 480
    depth shape:       640 x 480
    camera intrinsics: DEFAULT_CAMERA_INTRINSICS (hardcoded below)
    camera extrinsics: DEFAULT_CAMERA_EXTRINSICS (hardcoded below)

Usage
-----
    uvicorn galaxea_mujoco.runtime_perception.remote_perception_server:app \
        --host 0.0.0.0 --port 8088
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger("remote_perception_server")

# ---------------------------------------------------------------------------
# Path setup  --  add galaxea_mujoco/ to sys.path so local imports work
# ---------------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # galaxea_mujoco/
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from runtime_perception.warnings_filter import suppress_grounded_sam2_warnings

suppress_grounded_sam2_warnings()

from runtime_perception.backends.grounded_sam2_backend import GroundedSAM2Segmenter
from runtime_perception.backends.pointcloud_backend import (
    DEFAULT_CAMERA_EXTRINSICS,
    DEFAULT_CAMERA_INTRINSICS,
    PointCloudGenerator,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GROUNDED_SAM2_ROOT = str(PACKAGE_ROOT.parent / "Grounded-SAM-2")
"""Path to the Grounded-SAM-2 repository root."""

EXPECTED_RGB_SHAPE = (480, 640)
EXPECTED_DEPTH_SHAPE = (480, 640)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Galaxea Remote Perception Server",
    description="Remote GroundedSAM2 segmentation + point-cloud position via HTTP",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global model instances (loaded once at startup)
# ---------------------------------------------------------------------------

segmenter: GroundedSAM2Segmenter | None = None
pcd_generator: PointCloudGenerator | None = None


def _load_models() -> None:
    """Initialize GroundedSAM2Segmenter and PointCloudGenerator.

    This is called once at server startup so every /detect_pose request
    only does inference, not model-loading.
    """
    global segmenter, pcd_generator

    logger.info("Loading GroundedSAM2Segmenter ...")
    segmenter = GroundedSAM2Segmenter(
        grounded_sam2_root=GROUNDED_SAM2_ROOT,
        box_threshold=0.2,
        text_threshold=0.2,
        device=None,
        multimask_output=True,
    )
    segmenter._load()  # force eager loading of all sub-models (SAM2, DINO, BERT)
    logger.info("GroundedSAM2Segmenter loaded successfully")

    logger.info("Creating PointCloudGenerator ...")
    pcd_generator = PointCloudGenerator(
        camera_intrinsics=DEFAULT_CAMERA_INTRINSICS,
        camera_extrinsics=DEFAULT_CAMERA_EXTRINSICS,
        save_point_cloud=False,
        denoise=True,
        denoise_neighbors=10,
        denoise_std_ratio=5.0,
        use_dbscan=False,
    )
    logger.info("PointCloudGenerator ready")


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------


def _read_depth(path: Path) -> np.ndarray:
    """Read a uint16 depth PNG (or npy) and return a 2-D array."""
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Failed to read depth image: {path}")
    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth.squeeze(axis=2)
    if depth.ndim == 3:
        raise ValueError(f"Depth must be single-channel, got shape {depth.shape}")
    return depth


def _validate_depth(depth: np.ndarray) -> None:
    """Check that the depth array matches the expected protocol."""
    if depth.shape != EXPECTED_DEPTH_SHAPE:
        raise ValueError(
            f"Depth shape mismatch: got {depth.shape}, "
            f"expected {EXPECTED_DEPTH_SHAPE}"
        )
    if depth.dtype != np.uint16:
        raise ValueError(
            f"Depth dtype mismatch: got {depth.dtype}, expected uint16"
        )


def _validate_rgb(rgb: np.ndarray) -> None:
    """Check that the colour array matches the expected protocol."""
    if rgb.shape[:2] != EXPECTED_RGB_SHAPE:
        raise ValueError(
            f"RGB shape mismatch: got {rgb.shape[:2]}, "
            f"expected {EXPECTED_RGB_SHAPE}"
        )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@app.on_event("startup")
def startup() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting Galaxea Remote Perception Server ...")
    _load_models()
    logger.info("Server ready")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, Any]:
    """Lightweight health-check endpoint."""
    return {
        "status": "ok",
        "model_loaded": segmenter is not None and segmenter._loaded,
        "camera_intrinsics": DEFAULT_CAMERA_INTRINSICS,
        "camera_extrinsics": DEFAULT_CAMERA_EXTRINSICS,
        "expected_rgb_shape": list(EXPECTED_RGB_SHAPE),
        "expected_depth_shape": list(EXPECTED_DEPTH_SHAPE),
        "depth_unit": "uint16_mm",
    }


@app.post("/detect_pose")
async def detect_pose(
    rgb: UploadFile = File(...),
    depth: UploadFile = File(...),
    target_class: str = Form(...),
    coordinate_system: str = Form("world"),
) -> JSONResponse:
    """Detect *target_class* in *rgb* / *depth* and return its 3-D position.

    Accepts ``multipart/form-data``:

    - **rgb**         — JPEG or PNG colour image (will be saved as temp file)
    - **depth**       — uint16 PNG depth map (millimetres, 0 = invalid)
    - **target_class** — object name (e.g. ``\"red box\"``, ``\"apple\"``)
    - **coordinate_system** — ``\"camera\"`` | ``\"base\"`` | ``\"world\"`` (default ``\"world\"``)
    """
    # ---- validate inputs --------------------------------------------------
    if segmenter is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    target_class = str(target_class).strip().lower().rstrip(".")
    if not target_class:
        raise HTTPException(status_code=400, detail="target_class is empty")

    coordinate_system = coordinate_system.strip().lower()
    if coordinate_system not in ("camera", "base", "world"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid coordinate_system: {coordinate_system!r}. "
            f"Choose from: camera, base, world.",
        )

    await _log_upload_info(rgb, depth)

    # ---- save uploaded files & process ------------------------------------
    try:
        # Using TemporaryDirectory ensures cleanup even on exceptions.
        with tempfile.TemporaryDirectory(prefix="perception_") as tmpdir:
            tmp = Path(tmpdir)

            # Write RGB to disk (segment_image() reads from file path).
            rgb_bytes = await rgb.read()
            rgb_suffix = _infer_suffix(rgb, ".png")
            rgb_path = tmp / f"rgb{rgb_suffix}"
            rgb_path.write_bytes(rgb_bytes)

            # Write depth PNG to disk.
            depth_bytes = await depth.read()
            depth_path = tmp / "depth.png"
            depth_path.write_bytes(depth_bytes)

            # Read & validate depth array.
            depth_img = _read_depth(depth_path)
            _validate_depth(depth_img)

            # Read & validate RGB array.
            rgb_img = cv2.imread(str(rgb_path))
            if rgb_img is None:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to decode RGB image (sent as "
                    f"{rgb.filename})",
                )
            _validate_rgb(rgb_img)

            # ---- GroundedSAM2 segmentation --------------------------------
            seg_result = segmenter.segment_image(
                image_path=str(rgb_path),
                target_class=target_class,
                # Don't save mask/annotated files on the server by default;
                # the robot can request them later if needed.
            )
            if seg_result is None:
                logger.info("No detection for target_class=%s", target_class)
                return JSONResponse(
                    content={
                        "success": False,
                        "target_class": target_class,
                        "coordinate_system": coordinate_system,
                        "error": f"Grounded-SAM2 did not detect: {target_class}",
                    }
                )

            mask = seg_result.mask
            mask_pixel_count = int(np.count_nonzero(mask > 0))
            valid_depth_count = int(
                np.count_nonzero((mask > 0) & (depth_img > 0))
            )
            logger.info(
                "Detected %s: mask=%d px, valid_depth=%d px",
                target_class,
                mask_pixel_count,
                valid_depth_count,
            )

            # ---- point-cloud -> 3-D position ------------------------------
            pcd_result = pcd_generator.generate_point_cloud(
                color_image_aligned=rgb_img,
                depth_image_aligned=depth_img,
                mask=mask,
                downsample_scale=1.0,
                target_coordinate_system=coordinate_system,
            )

            if pcd_result.get("state") != "success":
                logger.warning(
                    "Point cloud failed for %s: %s",
                    target_class,
                    pcd_result.get("info"),
                )
                return JSONResponse(
                    content={
                        "success": False,
                        "target_class": target_class,
                        "coordinate_system": coordinate_system,
                        "mask_pixel_count": mask_pixel_count,
                        "valid_depth_count": valid_depth_count,
                        "error": f"Point cloud generation failed: "
                        f"{pcd_result.get('info')}",
                    }
                )

            # PointCloudGenerator returns mm; convert to metres.
            position_mm = [
                float(pcd_result["x"]),
                float(pcd_result["y"]),
                float(pcd_result["z"]),
            ]
            position_m = [round(v / 1000.0, 6) for v in position_mm]

            bbox_xyxy = (
                seg_result.candidate.get("xyxy")
                if seg_result.candidate
                else None
            )
            candidate_info = (
                {
                    "score": seg_result.candidate.get("score", 0.0),
                    "label": seg_result.candidate.get("label", ""),
                    "detector": seg_result.candidate.get("detector", ""),
                }
                if seg_result.candidate
                else {}
            )

            response = {
                "success": True,
                "target_class": target_class,
                "coordinate_system": coordinate_system,
                "position_m": position_m,
                "position_raw_mm": position_mm,
                "bbox_xyxy": bbox_xyxy,
                "mask_pixel_count": mask_pixel_count,
                "valid_depth_count": valid_depth_count,
                "depth_unit": "uint16_mm",
                "point_count": int(pcd_result.get("point_count", 0)),
                "candidate": candidate_info,
            }
            return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled error in /detect_pose")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {exc}",
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _log_upload_info(rgb: UploadFile, depth: UploadFile) -> None:
    """Log upload metadata (not the bytes)."""
    logger.info(
        "Received /detect_pose: rgb=%s (%d bytes), depth=%s (%d bytes)",
        rgb.filename,
        rgb.size or -1,
        depth.filename,
        depth.size or -1,
    )


def _infer_suffix(upload: UploadFile, fallback: str = ".png") -> str:
    """Return the file extension from the upload filename, or *fallback*."""
    if upload.filename:
        _, ext = Path(upload.filename).stem, Path(upload.filename).suffix
        if ext.lower() in (".jpg", ".jpeg", ".png"):
            return ext
    return fallback
