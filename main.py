import cv2
from Tracker import *
import numpy as np

tracker = Tracker()

cap = cv2.VideoCapture("traffic.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / fps)
frame_count = np.zeros(1000)

kernal_opening = np.ones((3, 3), np.uint8)
kernal_closing = np.ones((11, 11), np.uint8)
kernal_erode = np.ones((5, 5), np.uint8)
bgSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

while True:
    ret, frame = cap.read()

    if not ret:
        tracker.sumary()
        break
    
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    height, width, _ = frame.shape

    roi = frame[50:height, 200:width]

    fgMask = bgSubtractor.apply(roi)
    ret, thresh = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)
    mask1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernal_opening)
    mask2 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernal_closing)
    binary_img = cv2.erode(mask2, kernal_erode)

    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    vehicles = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
            vehicles.append([x, y, w, h])

    objects_array = tracker.track(vehicles)

    for item in objects_array:
        x, y, w, h, id = item

        if (y >= 235 and y <= 430):
            frame_count[id] += 1

        if (y < 235):
            tracker.frame_count[id] = frame_count[id]

        v = tracker.calcSpeed(id)
        v = int(v * 3.6)

        if(v < tracker.getLimitSpeed()):
            cv2.putText(roi, "Id " + str(id) + ", km/h: " + str(v), (x, y-15), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,0), 1)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
        else:
            cv2.putText(roi, "Id " + str(id) + ", km/h: " + str(v), (x, y-15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 165, 255), 3)

        if (y < 235 and v != 0):
            tracker.capture(roi, x, y, h, w, v, id)

    cv2.line(roi, (0, 430), (960, 430), (0, 0, 255), 2)
    cv2.line(roi, (0, 235), (960, 235), (0, 0, 255), 2)

    cv2.imshow("ROI", roi)

    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        tracker.sumary()
        break

cap.release()
cv2.destroyAllWindows()
