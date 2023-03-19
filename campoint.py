import cv2
import numpy as np

vid = cv2.VideoCapture(0)
show_debug = {
    'cap': False
}
markers = [
    cv2.imread('markers/TL.png'),
    cv2.imread('markers/TR.png'),
    cv2.imread('markers/BR.png'),
    cv2.imread('markers/BL.png'),
]

def cap():
    """Capture a video frame (interface to vid.read()).

    Returns:
        bool: success
        frame: frame captured
    """
    ret, frame = vid.read()
    if show_debug['cap']:
        cv2.imshow('frame', frame)
        cv2.waitKey(1) & 0xFF
    return ret, frame

def find_marker(n):
    capture = cap()[1]
    """Find nth marker on the video feed.

    Args:
        n (int): the ID of the marker to find.
    """
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(markers[n], None)
    kp2, des2 = orb.detectAndCompute(capture, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, m2 in matches:
        if m.distance<0.8*m2.distance:
            good.append(m)
    print(len(good))

    capkp = capture
    mkp = markers[n]
    capkp = cv2.drawKeypoints(capkp, kp2, None)
    mkp = cv2.drawKeypoints(mkp, kp1, None)

    cv2.imshow('capkp', capkp)
    cv2.imshow('mkp', mkp)

    img_features = cv2.drawMatches(markers[n], kp1, capture, kp2, good, None, flags=2)
    cv2.imshow('marker'+str(n), img_features)

def debug_show_markers():
    for i in range(len(markers)):
        cv2.imshow('marker'+str(i), markers[i])

def release():
    vid.release()