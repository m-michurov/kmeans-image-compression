from pathlib import Path
from typing import Any

import click
from PIL import Image
import numpy as np
from sklearn.cluster import MiniBatchKMeans


@click.command()
@click.argument('image_path', type=Path)
@click.argument('n_colors', type=int)
@click.option(
    '--out',
    required=False,
    type=Path
)
def main(
        image_path: Path,
        n_colors: int,
        out: Path | None = None
) -> None:
    assert n_colors > 0

    if out is None:
        out = Path(f'{image_path.stem}-{n_colors}{image_path.suffix}')

    pil_image: Any = Image.open(image_path).convert('RGB')
    image: np.ndarray = np.array(pil_image)

    height, width, channels = image.shape
    image = image.reshape(height * width, channels)

    # noinspection PyArgumentEqualDefault
    kmeans = MiniBatchKMeans(
        init='k-means++',
        n_clusters=n_colors
    )
    kmeans.fit(image)

    # Pixel -> Centroid
    compressed_image = kmeans.cluster_centers_[kmeans.labels_]
    compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)
    compressed_image = compressed_image.reshape((height, width, channels))

    Image.fromarray(compressed_image).save(out)


if __name__ == '__main__':
    main()
