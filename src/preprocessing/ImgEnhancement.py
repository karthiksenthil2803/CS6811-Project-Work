from pathlib import Path
import cv2
from utils import AGCWD_filter, Bilateral_Filter

def contrastAndSharpen(inputPath, outputPath):
    for imgFile in inputPath.glob('*'):
        if imgFile.is_file():
            imageName = imgFile.stem
            image = cv2.imread(str(imgFile))

        try:
            img_agcwd = AGCWD_filter.agcwd(image)
            img_bilateral = Bilateral_Filter.bilateral_filter(img_agcwd, 7, 20, 20)
            base_name, extension = imgFile.stem[:-2], imgFile.stem[-2:]
            newFileName = f"{base_name}_enh{extension}.png"
            outputFile = outputPath / newFileName
            cv2.imwrite(str(outputFile), img_bilateral)
            print(f"Enhanced image saved: {outputFile}")
        
        except Exception as e:
            print(f"Error processing {imageName} : {e}")