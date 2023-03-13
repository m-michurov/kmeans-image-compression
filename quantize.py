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


def quantize(
        image: Any,
        target_bpp: int,
        random_state: int | None = None
) -> Image.Image:
    assert 0 < target_bpp <= 8

    pixels: np.ndarray = np.array(image)

    height, width, channels = pixels.shape
    pixels = pixels.reshape((height * width, channels))

    # noinspection PyArgumentEqualDefault
    kmeans = MiniBatchKMeans(
        init='k-means++',
        n_clusters=2**target_bpp,
        random_state=random_state
    )
    kmeans.fit(pixels)

    pixels = kmeans.labels_.astype('uint8')
    result = Image.fromarray(pixels.reshape((height, width)), mode='P')
    result.putpalette(make_palette(color_space=kmeans.cluster_centers_))

    return result


@click.command()
@click.argument('image_path', type=Path)
@click.option(
    '--bpp',
    required=True,
    type=int,
    help='Must be in range [1, 8]'
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
