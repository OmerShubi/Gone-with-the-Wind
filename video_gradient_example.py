# https://www.geeksforgeeks.org/python-opencv-dense-optical-flow/
import cv2 as cv
import numpy as np


def display_motion(video_path, scale=0.5, show_rgb_motion=True, show_gray_motion=True):
    # The video feed is read in as
    # a VideoCapture object
    cap = cv.VideoCapture(video_path)

    # ret = a boolean return value from
    # getting the frame, first_frame = the
    # first frame in the entire video sequence
    _, first_frame = cap.read()

    first_frame = cv.resize(first_frame, dsize=(0, 0), fx=scale, fy=scale)
    # Converts frame to grayscale because we
    # only need the luminance channel for
    # detecting edges - less computationally
    # expensive
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    # Creates an image filled with zero
    # intensities with the same dimensions
    # as the frame
    mask = np.zeros_like(first_frame)

    # Sets image saturation to maximum
    mask[..., 1] = 255

    while cap.isOpened():

        # ret = a boolean return value from getting
        # the frame, frame = the current frame being
        # projected in the video
        _, frame = cap.read()
        if frame is None:
            print('End of Video.')
            break

        frame = cv.resize(frame, dsize=(0, 0), fx=scale, fy=scale)
        # frame = cv.resize(frame, dim)
        # Opens a new window and displays the input
        # frame
        cv.imshow("input", frame)

        # Converts each frame to grayscale - we previously
        # only converted the first frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Calculates dense optical flow by Farneback method
        flow = cv.calcOpticalFlowFarneback(prev=prev_gray, next=gray,
                                           flow=None, pyr_scale=0.5, levels=60, winsize=4, iterations=3,
                                           poly_n=7, poly_sigma=1.2, flags=0)

        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

        # Sets image hue according to the optical flow
        # direction
        mask[..., 0] = angle * 180 / np.pi / 2

        # Sets image value according to the optical flow
        # magnitude (normalized)
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

        # Converts HSV to RGB (BGR) color representation
        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
        h, s, v1 = cv.split(mask)

        # Opens a new window and displays the output frame
        if show_rgb_motion:
            cv.imshow("dense optical flow rgb", rgb)
        if show_gray_motion:
            cv.imshow("dense optical flow gray", v1)

        # Updates previous frame
        prev_gray = gray

        # Frames are read by intervals of 1 millisecond. The
        # programs breaks out of the while loop when the
        # user presses the 'q' key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # The following frees up resources and
    # closes all windows
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    print("Press 'q' to quit.")
    display_motion("P201229_070800_070900.avi")
