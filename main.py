"""
Hand Gesture Detection

This script detects hand gestures using a webcam or an image and displays the recognized gestures on the screen.

Author: Abhinav Ajit Menon

Library Used:
- CVZone: A Computer Vision package that makes it easy to run Image Processing and AI functions.
  At the core, it uses OpenCV and Mediapipe libraries.
  GitHub: https://github.com/cvzone/cvzone
"""

import cv2
import sys
from cvzone.HandTrackingModule import HandDetector

def detect_gestures(img):
    """
    Detect hand gestures in a given image and display the recognized gestures.
    """
    detector = HandDetector(detectionCon=0.8, maxHands=2)
    # Detect hands
    hands, img = detector.findHands(img, flipType=False)
    if hands:
        for hand in hands:
            # Check which fingers are up
            fingers = detector.fingersUp(hand)
            if fingers == [1, 1, 1, 1, 1]:
                gesture = "Open Palm"
            elif fingers == [0, 0, 0, 0, 0]:
                gesture = "Fist"
            elif fingers == [0, 0, 1, 0, 0]:
                gesture = "Middle Finger"
            elif fingers == [0, 0, 1, 1, 1]:
                gesture = "OK Sign"
            elif fingers == [0, 1, 1, 0, 0]:
                gesture = "V Sign"
            elif fingers == [1, 0, 0, 0, 0]:
                gesture = "Thumbs Up"
            else:
                gesture = "Other Gesture"
            # Get hand bounding box for displaying gesture text
            x, y, w, h = hand['bbox']
            cv2.putText(img, gesture, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return img

def main():
    mode = input("Enter 'webcam' for webcam mode or 'image' for image mode: ").strip().lower()
    if mode == 'webcam':
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)
        # detector = HandDetector(detectionCon=0.8, maxHands=2)
        while True:
            success, img = cap.read()
            if not success:
                break
            img = cv2.flip(img, 1)
            img = detect_gestures(img)
            cv2.imshow("Gesture Detection", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    elif mode == 'image':
        image_path = input("Enter the path to the image: ").strip()
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Image not found. Please provide a valid path.")
            sys.exit(1)
        img = detect_gestures(img)
        cv2.imshow("Gesture Detection - Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Invalid mode. Please enter either 'webcam' or 'image'.")


if __name__ == "__main__":
    main()
