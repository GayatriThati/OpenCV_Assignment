import cv2
import numpy as np

# a.Allow to load image from image gallery
img = cv2.imread('OpenCV_Assignment_Image.png')
image = img.copy()
ROI = img[50:350, 330:650]
gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
gray_blurred = cv2.blur(gray, (3, 3))

# b.Identifying the circle objects (wood log) from the image and Highlight using green border
edges = cv2.Canny(gray, 60, 200, apertureSize=3)
contours, hierarchy = cv2.findContours(edges,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
if contours is not None:
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 3500:
            cv2.drawContours(ROI, contours, -1, (0, 255, 0), 2)

detected_circles = cv2.HoughCircles(gray_blurred,
                                    cv2.HOUGH_GRADIENT, minDist=12, dp=30, param1=50,
                                    param2=30, minRadius=20, maxRadius=30)
cv2.imshow('Wood_logs', img)

# c.Identify the rectangle objects and highlight it by using blue rectangle
Rect_ROI = image[50:350, 150:800]
gray_r = cv2.cvtColor(Rect_ROI, cv2.COLOR_BGR2GRAY)
gray_blurred_r = cv2.blur(gray_r, (3, 3))
ret, thresh1 = cv2.threshold(gray_blurred_r, 140, 255, cv2.THRESH_BINARY_INV)
edges_r = cv2.Canny(thresh1, 60, 200, apertureSize=3)
lines = cv2.HoughLinesP(edges_r, 1, np.pi, 20, minLineLength=60, maxLineGap=10)
l = lines.shape[0]
for i in range(l):
    for x1, y1, x2, y2 in lines[i]:
        a = 10
        start_pt = (x1 - a, y1 - 100)
        end_pt = (x2 + a, y2 + 130)
        cv2.rectangle(Rect_ROI, start_pt, end_pt, (255, 0, 0), 2)
cv2.imshow('Rectangles', image)

# d. Identify the number of circle objects
if detected_circles is not None:
    detected_circles = np.uint16(np.around(detected_circles))
    count = 0
    for i in detected_circles[0, :]:
        count = count + 1
    print('Number of circles detected:', count)

cv2.waitKey(0)