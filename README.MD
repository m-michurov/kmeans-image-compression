# K-means++ for Image Compression Demo

This program uses k-means++ to reduce number of colors used in
image.
This can lead to image size reduction when using lossless formats
(such as `.png`).


## Usage
    Usage: compress.py [OPTIONS] IMAGE_PATH N_COLORS
    
    Options:
      --out PATH
      --help      Show this message and exit.

## Example

Input: 

[`radio.png`](./radio.png) (366 kB)

![radio.png](./radio.png)

Compress using:
```bash
py compress.py radio.png 64
```

Output:

[`radio-64.png`](./radio-64.png) (219 kB)

![radio-64.png](./radio-64.png)