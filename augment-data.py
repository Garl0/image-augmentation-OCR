import argparse
from glob import glob
import cv2
import sys
import os
import numpy as np
from deskew import determine_skew

def process(wd:str, bmin:int, bmax:int, inverse:bool, output) -> None:
    '''
        Process images and save to ./images
    '''
    os.chdir(wd)
    images = glob('*.jpg') + glob('*.jpeg') + glob('*.png')

    if not images:
        print(f'\nNothing to process at | {os.getcwd()} |\nUse -d or --dir (directory_path)')
        return
    
    if not os.path.isdir(f'./{output}'):
        os.mkdir(f'./{output}')

    print(f"\nFound {len(images)} images...")
    
    for image in images:
        print(f"Processing {image}...")
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (h, w) = gray.shape
        angle = determine_skew(img)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, bmin, bmax, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if inverse: binary = cv2.bitwise_not(binary)
        
        kernel = np.ones((3,3), np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=1)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        name, ext = image.rpartition('.')[0::2]
        
        cv2.imwrite(f"./{output}/{name}-1.{ext}", binary)
        cv2.imwrite(f"./{output}/{name}-2.{ext}", eroded)
        cv2.imwrite(f"./{output}/{name}-3.{ext}", dilated)
        cv2.imwrite(f"./{output}/{name}-4.{ext}", opened)
        cv2.imwrite(f"./{output}/{name}-5.{ext}", closed)

    print(f"Saved at ./{output} directory...")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        prog="augment-data",
                        description="A tool to help augment data for Optical Character Recognition (OCR)",
                        epilog="This program will process only images of formats (JPG, JPEG, PNG)")
    parser.add_argument('-d', '--dir', type=str, default=os.getcwd(), \
                        help="set directory path for the images to augment")
    parser.add_argument('-min', type=int, default=0, \
                        help="set the minimum value for binarization (0-255)")
    parser.add_argument('-max', type=int, default=255, \
                        help="set the maximum value for binarization (0-255)")
    parser.add_argument('-inv', type=bool, default=False, \
                        help="set the texts to white and background to black")
    parser.add_argument('-out', type=str, default='images', \
                        help="set output folder name")
    args = parser.parse_args()

    process(args.dir, args.min, args.max, args.inv, args.out)

    
