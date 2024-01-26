# IMPORTS

## External Imports
import os
from pathlib import Path
import numpy as np
import cv2

## Custom Imports
from utils import AGCWD_filter, Bilateral_Filter

# PATH SPECIFICATION
inputPath = Path('data/raw/')
outputPath = Path('data/processed/')

# DATA LOADING AND PREPROCESSING
for imgFile in inputPath.glob('*'):
    if imgFile.is_file():
        imageName = imgFile.stem
        image = cv2.imread(str(imgFile))
    
    try:
        img_agcwd = AGCWD_filter.agcwd(image)
        img_bilateral = Bilateral_Filter.bilateral_filter(img_agcwd, 7, 20, 20)
        newFileName = f"{imageName}_enh.png"
        outputFile = outputPath / newFileName
        cv2.imwrite(str(outputFile), img_bilateral)
        print(f"Enhanced image saved: {outputFile}")
    
    except Exception as e:
        print(f"Error processing {imageName} : {e}")