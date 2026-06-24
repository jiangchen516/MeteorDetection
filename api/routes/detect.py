"""交互式参数调整 API."""

import time
from pathlib import Path

import cv2
import numpy as np
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
    """使用指定参数运行检测."""
    image_path = req.get("image_path")
    params = req.get("params", {})

    if not image_path:
        return JSONResponse({"error": "image_path required"}, status_code=400)

    path = Path(image_path)
    if not path.exists():
        return JSONResponse({"error": f"Image not found: {image_path}"}, status_code=404)

    try:
        from core.detection.line_detector import LineDetector
        from core.utils.image_io import auto_load, save_image

        # 创建自定义检测器
        detector = LineDetector(_config_loaded=True)
        detector.adaptive_block_size = params.get("block_size", 21)
        detector.adaptive_c = params.get("c", -2)
        detector.min_line_length = params.get("min_line_length", 15)
        detector.morph_kernel_size = params.get("morph", 3)
        # 覆盖 detect 方法中的参数
        detector._blur_kernel = params.get("blur", 5)

        # 加载图片
        image = auto_load(path)
        if image is None:
            return JSONResponse({"error": "Failed to load image"}, status_code=400)

        # 运行检测
        start_time = time.time()

        # 使用自定义参数运行检测
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blur_ksize = params.get("blur", 5)
        blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
        binary = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            params.get("block_size", 21),
            params.get("c", -2),
        )

        # 形态学开运算
        morph_ksize = params.get("morph", 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        detections = 0
        h, w = gray.shape[:2]
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            aspect = max(bw, bh) / max(min(bw, bh), 1)

            if area < params.get("min_area", 5):
                continue
            if aspect < params.get("min_aspect", 2.0):
                continue

            # 边缘过滤
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            cx, cy = centroids[i]
            dist_to_edge = min(cx, cy, w - cx, h - cy)
            if dist_to_edge < 5:
                continue

            detections += 1

        elapsed = time.time() - start_time

        # 生成标注图
        from core.utils.visualization import draw_detections
        from core.detection.line_detector import DetectionResult

        # 收集检测结果用于绘图
        det_results = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if area < params.get("min_area", 5) or aspect < params.get("min_aspect", 2.0):
                continue

            component_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            rect = cv2.minAreaRect(np.vstack(contours))
            (cx, cy), (rw, rh), angle = rect
            length = max(rw, rh)
            width = min(rw, rh)

            if length < params.get("min_line_length", 15) or width > 20:
                continue

            box = cv2.boxPoints(rect)
            box_pts = [(int(p[0]), int(p[1])) for p in box]
            dx_long = box_pts[2][0] - box_pts[1][0]
            dy_long = box_pts[2][1] - box_pts[1][1]
            x1, y1 = int(cx - dx_long / 2), int(cy - dy_long / 2)
            x2, y2 = int(cx + dx_long / 2), int(cy + dy_long / 2)

            mean_bright = float(gray[component_mask > 0].mean()) if area > 0 else 0

            det_results.append(DetectionResult(
                x1=x1, y1=y1, x2=x2, y2=y2,
                brightness=mean_bright,
                score=min(length / 100.0, 1.0),
                confidence=min(aspect / 10.0, 1.0),
                width=width,
                box_points=box_pts,
            ))

        # 保存标注图
        output_dir = Path("storage/results") / "tune"
        output_dir.mkdir(parents=True, exist_ok=True)
        annotated = draw_detections(image, det_results)
        result_path = output_dir / f"{path.stem}_detected.jpg"
        save_image(annotated, result_path)

        return {
            "detections": len(det_results),
            "time": elapsed,
            "result_image": str(result_path),
            "params": params,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
