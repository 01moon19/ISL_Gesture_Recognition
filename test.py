import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)  # Detect up to 2 hands
classifier = Classifier("C:/Users/mayan/OneDrive/Desktop/Sign-Language-detection/sign_language_detection_model_final.h5", "C:/Users/mayan/OneDrive/Desktop/Sign-Language-detection/labels.txt")
offset = 20
imgSize = 300
counter = 0

labels = ["E", "H", "L","O"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)  # Detect hands

    if hands:
        # If only one hand is detected
        if len(hands) == 1:
            hand = hands[0]
            x, y, w, h = hand['bbox']  # Get bounding box

            # Crop and process the hand for prediction
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Create a white background
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]  # Crop the hand image
            aspectRatio = h / w  # Calculate aspect ratio

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize

            # Make the prediction
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            label = labels[index % len(labels)]

            # Display the prediction and bounding box
            cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x + 200, y - offset + 50), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, label, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        # If two hands are detected, combine them into a single entity
        elif len(hands) == 2:
            hand1 = hands[0]
            hand2 = hands[1]

            # Get the bounding box for both hands
            x1, y1, w1, h1 = hand1['bbox']
            x2, y2, w2, h2 = hand2['bbox']

            # Calculate the smallest bounding box that can encompass both hands
            x = min(x1, x2)
            y = min(y1, y2)
            w = max(x1 + w1, x2 + w2) - x
            h = max(y1 + h1, y2 + h2) - y

            # Crop and process the combined hand region for prediction
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Create a white background
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]  # Crop the region
            aspectRatio = h / w  # Calculate aspect ratio

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize

            # Make the prediction
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            label = labels[index % len(labels)]

            # Display the prediction and bounding box
            cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x + 200, y - offset + 50), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, label, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        # Show the cropped hand region and the result
        cv2.imshow('ImageWhite', imgWhite)

    # Show the output image with the prediction
    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)
