import cv2 as cv

# Reading images
# img = cv.imread('Resources/Photos/cat.jpg')

# cv.imshow('Cat', img)

# # This a keyboard binding function that waits for a specific delay or time in milliseconds for a key to be pressed. Here, if you press 0, fn will wait for an infinite amount of time for a keyboard key to be pressed

# cv.waitKey(0)

# Reading videos
capture = cv.VideoCapture('Resources/Videos/dog.mp4')

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
cv.waitKey(0)
