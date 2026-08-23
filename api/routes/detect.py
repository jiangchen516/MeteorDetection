"""交互式参数调整 API.

复用检测引擎 LineDetector，保证调参页面与主检测链路使用同一段逻辑
（单一逻辑来源，避免两处实现漂移）。
"""

import time
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/images/")
async def list_images():
    """列出可用的图片."""
    from core.utils.config import get_path

    upload_dir = get_path("storage_dir") / "uploads"
    images = []
    if upload_dir.exists():
        for f in sorted(upload_dir.iterdir()):
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'):
                images.append({
                    "name": f.name,
                    "path": str(f),
                })
    return {"images": images}


@router.post("/")
async def detect_with_params(req: dict):
    """使用指定参数运行检测.

    前端传入的参数会被注入到与 worker 同款的 LineDetector 实例上，
    因此调参结果即为线上检测逻辑。
    """
    image_path = req.get("image_path")
    params = req.get("params") or {}

    if not image_path:
        return JSONResponse({"error": "image_path required"}, status_code=400)

    # 将 URL 路径转换为绝对路径
    if str(image_path).startswith('/'):
        relative_path = str(image_path)[1:]
        path = Path(relative_path)
    else:
        path = Path(image_path)

    if not path.exists():
        return JSONResponse(
            {"error": f"Image not found: {image_path} (resolved: {path})"},
            status_code=404,
        )

    try:
        from core.detection.line_detector import LineDetector
        from core.utils.config import get_section
        from core.utils.image_io import auto_load, save_image
        from core.utils.visualization import draw_detections

        # 构建与主检测链路完全一致的检测器，仅覆盖前端传入的参数
        detector = LineDetector(_config_loaded=True)
        # 评分阈值从配置读取，保证调参结果与 worker 主链路一致
        detector.score_threshold = float(
            get_section("meteor_detection").get(
                "score_threshold", detector.score_threshold))
        detector.adaptive_block_size = int(
            params.get("block_size", detector.adaptive_block_size))
        detector.adaptive_c = float(params.get("c", detector.adaptive_c))
        detector.min_line_length = int(
            params.get("min_line_length", detector.min_line_length))
        detector.morph_kernel_size = int(
            params.get("morph", detector.morph_kernel_size))
        detector.close_kernel_size = int(
            params.get("close", detector.close_kernel_size))
        detector.blur_kernel_size = int(
            params.get("blur", detector.blur_kernel_size))
        detector.min_area = int(params.get("min_area", detector.min_area))
        detector.min_aspect = float(
            params.get("min_aspect", detector.min_aspect))

        # 加载图片
        image = auto_load(path)
        if image is None:
            return JSONResponse({"error": "Failed to load image"}, status_code=400)

        # 运行检测（单一逻辑来源）
        start_time = time.time()
        detections = detector.detect(image)
        elapsed = time.time() - start_time

        # 保存标注图
        output_dir = Path("storage/results") / "tune"
        output_dir.mkdir(parents=True, exist_ok=True)
        annotated = draw_detections(image, detections)
        result_path = output_dir / f"{path.stem}_detected.jpg"
        save_image(annotated, result_path)

        return {
            "detections": len(detections),
            "time": elapsed,
            "result_image": str(result_path),
            "params": params,
            "results": [
                {
                    "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
                    "brightness": d.brightness, "confidence": d.confidence,
                    "score": d.score, "width": getattr(d, "width", 0.0),
                    "object_class": getattr(d, "object_class", "unknown"),
                }
                for d in detections
            ],
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
