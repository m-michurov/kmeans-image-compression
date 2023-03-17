from pathlib import Path
from typing import Any

import click
from PIL import Image
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def make_palette(color_space: np.ndarray) -> list[int]:
    color_space = np.clip(color_space, 0, 255).astype('uint8')

    colors, channels = color_space.shape
    palette_data = list(color_space.reshape((channels * colors,)))

    palette = (palette_data * (256 // colors))

    return palette


def compress_color_space(
        pixels: np.ndarray,
        target_bpp: int,
        random_state: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    # noinspection PyArgumentEqualDefault
    kmeans = MiniBatchKMeans(
        init='k-means++',
        n_clusters=2 ** target_bpp,
        random_state=random_state
    )
    kmeans.fit(pixels)

    labels = kmeans.labels_.astype('uint8') if target_bpp <= 8 else kmeans.labels_
    color_palette = np.clip(kmeans.cluster_centers_, 0, 255).astype('uint8')

    return color_palette, labels


def quantize(
        image: Any,
        target_bpp: int,
        random_state: int | None = None
) -> Image.Image:
    assert 0 < target_bpp <= 24

    pixels: np.ndarray = np.array(image)

    height, width, channels = pixels.shape
    pixels = pixels.reshape((height * width, channels))

    color_palette, labels = compress_color_space(pixels, target_bpp, random_state)

    if target_bpp > 8:
        return Image.fromarray(color_palette[labels].reshape((height, width, channels)), mode='RGB')

    result = Image.fromarray(labels.reshape((height, width)), mode='P')
    result.putpalette(make_palette(color_palette))

    return result


@click.command()
@click.argument('image_path', type=Path)
@click.option(
    '--bpp',
    required=True,
    type=int,
    help='Must be in range [1, 24]'
)
@click.option(
    '--out',
    required=False,
    type=Path
)
@click.option(
    '--random-state',
    required=False,
    type=int
)
def main(
        image_path: Path,
        bpp: int,
        out: Path | None = None,
        random_state: int | None = None
) -> None:
    if out is None:
        out = Path(f'{image_path.stem}-{2 ** bpp}{image_path.suffix}')

    with Image.open(image_path).convert('RGB') as source_image:
        with quantize(source_image, bpp, random_state) as compressed_image:
            try:
                compressed_image.save(out)
            except OSError:
                compressed_image.convert('RGB').save(out)


if __name__ == '__main__':
    main()
