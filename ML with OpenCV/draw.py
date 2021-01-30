import cv2 as cv
import numpy as np

# Create a blank image
blank = np.zeros((500, 500, 3), dtype='uint8')
cv.imshow('Black', blank)

# 1. Paint the pixels of this image
# blank[200:400, 100:300] = 0, 255, 0
# cv.imshow('Green', blank)

# 2. Draw a rectangle
cv.rectangle(blank, (0, 0),
             (blank.shape[1]//2, blank.shape[0]//2), (0, 255, 0), thickness=2)
cv.imshow('Green Rect', blank)

# 3. Draw a circle
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2),
          20, (0, 0, 255), thickness=-1)
cv.imshow('Red Circle', blank)

# 4. Draw a line
cv.line(blank, (0, 0),
        (blank.shape[1]//2, blank.shape[0]//2), (255, 255, 255), thickness=3)
cv.imshow('Line', blank)

# 5. Write text
cv.putText(blank, 'Hello', (255, 255),
           cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
cv.imshow('text', blank)

cv.waitKey(0)
