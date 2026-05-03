"""实时检测模式 — 文件夹监听 + 持续检测."""

import time
from pathlib import Path
from typing import Callable

from core.pipeline import MeteorPipeline
from core.utils.config import get_section
from core.utils.image_io import is_supported_format
from core.utils.logging import get_logger

logger = get_logger(__name__)


class RealtimeDetector:
    """实时检测器 — 监听文件夹，新图片自动检测."""

    def __init__(
        self,
        watch_dir: str | Path,
        output_dir: str | Path,
        background_path: str | Path | None = None,
        poll_interval: float = 2.0,
        on_detection: Callable[[str, list[dict]], None] | None = None,
        on_meteor: Callable[[dict], None] | None = None,
    ):
        """
        Args:
            watch_dir: 监听目录.
            output_dir: 输出目录.
            background_path: 背景图像路径.
            poll_interval: 轮询间隔（秒）.
            on_detection: 检测完成回调 (image_path, results).
            on_meteor: 发现流星回调 (meteor_info).
        """
        self.watch_dir = Path(watch_dir)
        self.output_dir = Path(output_dir)
        self.background_path = background_path
        self.poll_interval = poll_interval
        self.on_detection = on_detection
        self.on_meteor = on_meteor

        self._running = False
        self._processed: set[str] = set()

    def start(self) -> None:
        """开始监听."""
        self._running = True
        logger.info(f"Starting realtime detection on {self.watch_dir}")
        logger.info(f"Poll interval: {self.poll_interval}s")

        bg = None
        if self.background_path:
            from core.utils.image_io import auto_load
            bg = auto_load(self.background_path)

        pipeline = MeteorPipeline(background_image=bg)

        try:
            while self._running:
                self._check_new_files(pipeline)
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            logger.info("Realtime detection stopped by user")
        finally:
            self._running = False

    def stop(self) -> None:
        """停止监听."""
        self._running = False
        logger.info("Realtime detection stopping...")

    def _check_new_files(self, pipeline: MeteorPipeline) -> None:
        """检查新文件并处理."""
        if not self.watch_dir.exists():
            return

        for f in sorted(self.watch_dir.iterdir()):
            if not f.is_file():
                continue
            if not is_supported_format(f):
                continue

            file_key = f"{f.name}_{f.stat().st_mtime}"
            if file_key in self._processed:
                continue

            # 等待文件写入完成
            size1 = f.stat().st_size
            time.sleep(0.5)
            size2 = f.stat().st_size
            if size1 != size2:
                continue  # 文件还在写入

            logger.info(f"New file detected: {f.name}")
            self._processed.add(file_key)

            try:
                results = pipeline.run_single(f, self.output_dir / f.stem)

                if self.on_detection:
                    self.on_detection(str(f), results)

                # 高置信度结果触发流星事件
                for r in results:
                    if r.get("confidence", 0) > 0.7:
                        if self.on_meteor:
                            self.on_meteor(r)
                        logger.info(
                            f"☄️ 流星事件: {f.name} "
                            f"({r['x1']},{r['y1']})-({r['x2']},{r['y2']}) "
                            f"置信度={r['confidence']:.2f}"
                        )

            except Exception as e:
                logger.error(f"Failed to process {f.name}: {e}")


def run_realtime(
    watch_dir: str | Path,
    output_dir: str | Path,
    background_path: str | Path | None = None,
    poll_interval: float = 2.0,
) -> None:
    """便捷函数：启动实时检测."""
    detector = RealtimeDetector(
        watch_dir=watch_dir,
        output_dir=output_dir,
        background_path=background_path,
        poll_interval=poll_interval,
    )
    detector.start()
