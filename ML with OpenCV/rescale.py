import cv2 as cv


def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


def changeRes(width, height):
    capture.set(3, width)
    capture.set(4, height)


# Rescaling images
img = cv.imread('./Resources/Photos/cat.jpg')


resized_img = rescaleFrame(img, 0.2)
cv.imshow('Cat', img)
cv.imshow('Cat resized', resized_img)

cv.waitKey(0)

# Rescaling videos
capture = cv.VideoCapture('./Resources/Videos/dog.mp4')

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)
    cv.imshow('Video resized', rescaleFrame(frame, 0.5))

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
capture.destroyAllWindows()
cv.waitKey(0)
