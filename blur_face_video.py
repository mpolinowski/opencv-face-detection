# USAGE
# python blur_face_video.py --url 'rtsp://admin:instar@192.168.2.19/livestream/12' --face face_detector --method simple
# python blur_face_video.py --url 'rtsp://admin:instar@192.168.2.19/livestream/11' --face face_detector --method
# pixelated

from pyimagesearch.face_blurring import anonymize_face_pixelate
from pyimagesearch.face_blurring import anonymize_face_simple
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# Parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--url", help="RTSP streaming URL", default="rtsp://admin:instar@192.168.2.19/livestream/13")
ap.add_argument("-f", "--face", required=True, help="path to detector model")
ap.add_argument("-m", "--method", type=str, default="simple", choices=["simple", "pixelated"], help="face blurring "
                                                                                                    "method")
ap.add_argument("-b", "--blocks", type=int, default=20, help="number of pixel blocks for pixelate")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability of positive detection")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000_fp16.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# initialize your internal webcam stream and allow the camera sensor to warm up
# print("[INFO] starting video stream")
# vs = VideoStream(src=0).start()
# time.sleep(2.0)

# Get video stream from IP camera
print("[INFO] starting video stream")
vs = VideoStream(args["url"]).start()

# Loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 1080 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=1080)
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the face detections
    net.setInput(blob)
    detections = net.forward()
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            # check to see if we are applying the "simple" face
            # blurring method
            if args["method"] == "simple":
                face = anonymize_face_simple(face, factor=3.0)
            # otherwise, we must be applying the "pixelated" face
            # anonymization method
            else:
                face = anonymize_face_pixelate(face,
                                               blocks=args["blocks"])
            # store the blurred face in the output image
            frame[startY:endY, startX:endX] = face

        # show the output frame
        cv2.imshow("Frame", frame)

    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        vs.stop()
        break


