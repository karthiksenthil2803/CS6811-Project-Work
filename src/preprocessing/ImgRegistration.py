import numpy as np
import imutils
import cv2

from utils import Align_Images

def registerImage(img_path, temp_path, maxFeatures = 21000):
    image = cv2.imread(str(img_path))
    template = cv2.imread(str(temp_path))
    
    # align the images
    print("[INFO] aligning images...")
    aligned = Align_Images.align_images(image=image,
                                        template=template,
                                        maxFeatures=maxFeatures,
                                        debug=True)
    # resize both the aligned and template images so we can easily
    # visualize them on our screen

    # aligned = imutils.resize(aligned, width=700)
    # template = imutils.resize(template, width=700)
    # # our first output visualization of the image alignment will be a
    # # side-by-side comparison of the output aligned image and the
    # # template
    # stacked = np.hstack([aligned, template])
    # # our second image alignment visualization will be *overlaying* the
    # # aligned image on the template, that way we can obtain an idea of
    # # how good our image alignment is
    # overlay = template.copy()
    # output = aligned.copy()
    # cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
    # # show the two output image alignment visualizations
    # cv2.imshow("Image Alignment Stacked.jpg", stacked)
    # cv2.imshow("new_aligned.jpg", aligned)
    # cv2.imshow("Image Alignment Overlay.jpg", output)
    # cv2.waitKey(0)

    return aligned