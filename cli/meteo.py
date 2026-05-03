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


if __name__ == "__main__":
    cli()
