# K-means++ for Image Quantization Demo

This program uses k-means++ to compress image color space.
When possible, image will be saved with 8-bit color using 
computed color clusters as color palette.

Although supported bpp values are form 1 to 8,
resulting image will always have at least 8bpp. Still, lower
bpp values (smaller color space) can lead to lower image 
sizes due to compression applied when saving in given format.


## Usage

    Usage: quantize.py [OPTIONS] IMAGE_PATH
    
    Options:
      --bpp INTEGER           Must be in range [1, 8]  [required]
      --out PATH
      --random-state INTEGER
      --help                  Show this message and exit.

## Example

Input: 

[`radio.png`](./radio.png) (366 kB)

![radio.png](./radio.png)

Compress using:
```bash
py quantize.py radio.png --bpp 6 --random-state=42
```

Output:

[`radio-64.png`](./radio-64.png) (123 kB)

![radio-64.png](./radio-64.png)