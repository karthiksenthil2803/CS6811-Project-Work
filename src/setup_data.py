# IMPORTS

## External Imports
import os
from pathlib import Path
import numpy as np
import cv2

## Custom Imports
from utils import AGCWD_filter, Bilateral_Filter, Align_Images
from preprocessing import ImgEnhancement, ImgRegistration

# PATH SPECIFICATION
inputPathA = Path('data/raw/A/')
outputPathA = Path('data/processed/A/')
inputPathB = Path('data/raw/B/')
outputPathB = Path('data/processed/B/')

# # DATA LOADING AND PREPROCESSING

# ImgEnhancement.contrastAndSharpen(inputPathA, outputPathA)
# ImgEnhancement.contrastAndSharpen(inputPathB, outputPathB)

# PATH SPECIFICATION
inputPathA = Path('data/processed/A/')
outputPathA = Path('data/final/A/')
inputPathB = Path('data/processed/B/')
outputPathB = Path('data/final/B/')

# # IMAGE REGISTRATION

for imgFileA in inputPathA.glob('*'):
    if imgFileA.is_file():
        imageBaseName = imgFileA.stem[:-2]
        print(imageBaseName)
        imgFileB = inputPathB / f"{imageBaseName}_B.png"
        
    if imgFileA.is_file() and imgFileB.is_file():
        try:
            modified_B = cv2.imread(str(imgFileB))
            # modified_A = ImgRegistration.registerImage_using_Manual_Feature_Extraction(imgFileA, imgFileB, 27500)
            modified_A = ImgRegistration.registerImage_using_Deep_Feature_Extraction(imgFileA, imgFileB)
            newFileNameA = f"{imageBaseName}_fin_A.png"
            newFileNameB = f"{imageBaseName}_fin_B.png"
            outputFileA = outputPathA / newFileNameA
            outputFileB = outputPathB / newFileNameB
            cv2.imwrite(str(outputFileA),modified_A)
            cv2.imwrite(str(outputFileB),modified_B)
            # print('Registration is Over')
            
        except Exception as e:
            print(f"Error processsing {imageBaseName} : {e}")