# Image Augmentation tool for OCR Training

This tool will do data augmentation techniques to turn raw images into machine-readable data for better accuracy for OCR training.

## Installation:

use `pip install -r requirements.txt`
To install all the dependencies for the tool.

## Usage:
`python augment-data.py`

With the following parameters for custom setting:

```
-h | --help  display help info
-d | --dir   set directory path for the images to augment
-min         set the minimum value for binarization (0-255)
-max         set the maximum value for binarization (0-255)
-inv         set the texts to white and background to black
-out         set output folder name
```

# Examples:
`python augment-data.py -d ./raw-images -min 0 -max 255 -out images`

`python augment-data.py -d ./raw-images -min 0 -max 255 -out inverted -inv True`
