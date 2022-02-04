# USAGE
# python blur_face.py --image resources/metro.jpg --face face_detector --method simple
# python blur_face.py --image resources/metro.jpg --face face_detector --method pixelated

from pyimagesearch.face_blurring import anonymize_face_pixelate
from pyimagesearch.face_blurring import anonymize_face_simple
import numpy as np
import argparse
import cv2
import os

# Parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to image")
ap.add_argument("-f", "--face", required=True, help="path to detector model")
ap.add_argument("-m", "--method", type=str, default="simple", choices=["simple", "pixelated"], help="face blurring "
                                                                                                    "method")
ap.add_argument("-b", "--blocks", type=int, default=12, help="number of pixel blocks for pixelate")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability of positive detection")
args = vars(ap.parse_args())

# Load serialized face detector from disk
print("[INFO] loading face detector model")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000_fp16.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# Copy resource image and get dimensions
source_image = cv2.imread(args["image"])
image = source_image.copy()
(h, w) = image.shape[:2]

# Get blob from image
blob = cv2.dnn.blobFromImage(source_image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# Get face detections from blob
print("[INFO] computing face detections")
net.setInput(blob)
detections = net.forward()

# Loop over detections
for i in range(0, detections.shape[2]):
    # extract confidence from detections
    confidence = detections[0, 0, i, 2]
    # filter by min confidence
    if confidence > args["confidence"]:
        # compute bounding box for
        # passing detections
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # extract face ROI
        face = source_image[startY:endY, startX:endX]
        # select blur method
        if args["method"] == "simple":
            face = anonymize_face_simple(face, factor=3.0)
        else:
            face = anonymize_face_pixelate(face, blocks=args["blocks"])

        # store blurred face in output image
        source_image[startY:endY, startX:endX] = face

# display results
output = np.hstack([image, source_image])
cv2.imshow("Output", output)
cv2.imwrite("processed/face_blur.jpg", source_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



















