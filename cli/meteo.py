"""CLI 入口 — 流星探测命令行工具."""

import click
from pathlib import Path

from core import load_config, setup_logging
from core.pipeline import MeteorPipeline
from core.preprocessing.fisheye import rectify_fisheye
from core.preprocessing.merge import merge_images
from core.utils.logging import get_logger

logger = get_logger(__name__)


@click.group()
def cli():
    """流星探测工具."""
    setup_logging()
    load_config()


@cli.command()
@click.argument("image", type=click.Path(exists=True))
@click.option("--background", "-b", type=click.Path(exists=True), default=None,
              help="背景图像路径")
@click.option("--output", "-o", type=click.Path(), default="storage/results",
              help="输出目录")
def detect_single(image, background, output):
    """检测单张图像中的流星."""
    pipeline = MeteorPipeline(background_path=background)
    results = pipeline.run_single(image, output)
    if results:
        click.echo(f"检测到 {len(results)} 个候选流星:")
        for r in results:
            click.echo(
                f"  ({r['x1']},{r['y1']})-({r['x2']},{r['y2']}) "
                f"置信度={r['confidence']:.2f} 亮度={r['brightness']:.0f}"
            )
    else:
        click.echo("未检测到候选流星")


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--background", "-b", type=click.Path(exists=True), default=None,
              help="背景图像路径")
@click.option("--output", "-o", type=click.Path(), default="storage/results",
              help="输出目录")
def detect_batch(directory, background, output):
    """批量检测目录中的图像."""
    pipeline = MeteorPipeline(background_path=background)
    results = pipeline.run_batch(directory, output)
    if results:
        click.echo(f"共检测到 {len(results)} 个候选流星")
    else:
        click.echo("未检测到候选流星")


@cli.command()
@click.argument("input", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
def rectify(input, output):
    """校正鱼眼图像."""
    rectify_fisheye(input, output)
    click.echo(f"已保存到 {output}")


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.argument("output", type=click.Path())
def merge(directory, output):
    """叠加目录中的图像."""
    merge_images(directory, output)
    click.echo(f"已保存到 {output}")


@cli.command("fetch-tle")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="TLE 文件输出目录（默认 outputs/，与 SatelliteFilter 一致）")
@click.option("--group", multiple=True,
              default=["visual", "stations", "weather", "amateur"],
              help="从 CelesTrak 下载的分组（可多次指定），默认常用四组")
@click.option("--if-older", type=int, default=7,
              help="仅当本地文件缺失或早于 N 天时下载，默认 7 天")
@click.option("--no-smoke", is_flag=True, help="下载后跳过卫星比对冒烟测试")
def fetch_tle(output, group, if_older, no_smoke):
    """从 CelesTrak 下载卫星 TLE 数据到 outputs/（供 SatelliteFilter 使用）.

    TLE 有效期约几天到几周，建议定期运行（如 --if-older 7）。
    """
    from core.utils.config import get_path
    from datetime import datetime

    out_dir = Path(output) if output else get_path("outputs_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().timestamp()

    tmp_root = out_dir / ".tle_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    downloaded = []
    skipped = []
    for g in group:
        dest = out_dir / f"TLE_{g}.txt"
        # 判断是否需要下载
        need = True
        if dest.exists() and if_older > 0:
            age_days = (now - dest.stat().st_mtime) / 86400
            if age_days < if_older:
                need = False
                skipped.append(f"{g} ({age_days:.1f}天前)")
        if not need:
            continue

        url = f"https://celestrak.com/NORAD/elements/{g}.txt"
        click.echo(f"Downloading {url} ...")
        try:
            import requests
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            # 先写临时文件，成功后原子替换，避免留下残缺文件
            tmp_file = tmp_root / f"TLE_{g}.txt"
            tmp_file.write_bytes(resp.content)
            # 简单校验：至少应含 2 行 TLE 数据（首行卫星名）
            lines = [ln for ln in resp.text.strip().splitlines() if ln.strip()]
            if len(lines) < 1:
                raise RuntimeError("返回内容为空或格式异常")
            tmp_file.replace(dest)
            downloaded.append(f"{g} ({len(lines)} 行)")
        except Exception as e:
            click.echo(f"  ⚠ 下载失败 {g}: {e}", err=True)

    # 清理临时目录
    try:
        tmp_root.rmdir()
    except OSError:
        pass

    if downloaded:
        click.echo("已下载: " + ", ".join(downloaded))
    if skipped:
        click.echo("跳过（足够新）: " + ", ".join(skipped))
    if not downloaded and not skipped:
        click.echo("无可执行操作。")

    # 冒烟：用刚下载的 TLE 验证 SatelliteFilter 能正常比对
    if not no_smoke and downloaded:
        from core.classification.satellite_filter import SatelliteFilter
        filter_ = SatelliteFilter(tle_dir=out_dir)
        try:
            tle_files = sorted(out_dir.glob("TLE*.txt"))
            if tle_files:
                import skyfield.api as sfa
                with sfa.load.open(str(tle_files[0])) as f:
                    n_sats = len(sfa.load.tle_file(f))
                likely, reasons = filter_.is_likely_satellite(
                    datetime.now(), velocity=30.0, linearity=0.99,
                )
                click.echo(
                    f"卫星比对冒烟: {tle_files[0].name} 含 {n_sats} 颗卫星, "
                    f"is_likely_satellite={likely}, 原因={reasons or ['无']}"
                )
        except Exception as e:
            click.echo(f"⚠ 卫星比对冒烟未完成: {e}", err=True)
        else:
            click.echo("卫星比对冒烟完成。")


if __name__ == "__main__":
    cli()
