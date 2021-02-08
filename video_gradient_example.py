# https://www.geeksforgeeks.org/python-opencv-dense-optical-flow/
import cv2 as cv
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import os
import pandas as pd

import seaborn as sns
sns.set_theme(color_codes=True)


def display_motion(video_path, resize_scale=0.5, levels=60, winsize=4, show_rgb_motion=True, show_gray_motion=True):
    # The video feed is read in as
    # a VideoCapture object
    cap = cv.VideoCapture(video_path)

    # ret = a boolean return value from
    # getting the frame, first_frame = the
    # first frame in the entire video sequence
    _, first_frame = cap.read()

    first_frame = cv.resize(first_frame, dsize=(0, 0), fx=resize_scale, fy=resize_scale)
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

        frame = cv.resize(frame, dsize=(0, 0), fx=resize_scale, fy=resize_scale)
        # frame = cv.resize(frame, dim)
        # Opens a new window and displays the input
        # frame
        cv.imshow("input", frame)

        # Converts each frame to grayscale - we previously
        # only converted the first frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Calculates dense optical flow by Farneback method
        flow = cv.calcOpticalFlowFarneback(prev=prev_gray, next=gray,
                                           flow=None, pyr_scale=0.5, levels=levels, winsize=winsize, iterations=3,
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


def get_stats(video_path, resize_scale=0.5, levels=60, winsize=4, show_rgb_motion=True, show_gray_motion=True,
              num_frames=20):
    # The video feed is read in as
    # a VideoCapture object
    cap = cv.VideoCapture(video_path)

    # ret = a boolean return value from
    # getting the frame, first_frame = the
    # first frame in the entire video sequence
    _, first_frame = cap.read()

    first_frame = cv.resize(first_frame, dsize=(0, 0), fx=resize_scale, fy=resize_scale)
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
    index = 1
    return_val_full = 0
    return_val_cut = 0

    while cap.isOpened():

        # ret = a boolean return value from getting
        # the frame, frame = the current frame being
        # projected in the video
        _, frame = cap.read()
        if frame is None:
            print('End of Video.')
            break

        frame = cv.resize(frame, dsize=(0, 0), fx=resize_scale, fy=resize_scale)
        # frame = cv.resize(frame, dim)
        # Opens a new window and displays the input
        # frame
        if show_gray_motion or show_rgb_motion:
            cv.imshow("input", frame)

        # Converts each frame to grayscale - we previously
        # only converted the first frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Calculates dense optical flow by Farneback method
        flow = cv.calcOpticalFlowFarneback(prev=prev_gray, next=gray,
                                           flow=None, pyr_scale=0.5, levels=levels, winsize=winsize, iterations=3,
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
        v1_cut = v1[250:700, 600:1100]
        # Opens a new window and displays the output frame
        if show_rgb_motion or show_gray_motion:
            cv.waitKey(1)
        if show_rgb_motion:
            cv.imshow("dense optical flow rgb", rgb)
        if show_gray_motion:
            cv.imshow("dense optical flow gray", v1)
        if show_gray_motion:
            cv.imshow("dense optical flow gray cutout", v1_cut)

        if index != 1:
            return_val_full += np.mean(v1)
            return_val_cut += np.max(v1_cut)

        if index == num_frames:
            cap.release()
            return return_val_full / (index - 1), return_val_cut / (index - 1)

        index += 1
        # Updates previous frame
        prev_gray = gray

    # The following frees up resources and
    # closes all windows
    cap.release()
    cv.destroyAllWindows()
    return return_val_full / (index - 1), return_val_cut / (index - 1)


"""
top left: 600, 250
top right: 1100, 250
bottom: 700
"""


def get_paths(video_path):
    # define the path
    currentDirectory = pathlib.Path(video_path)
    # define the pattern
    currentPattern = "*.avi"
    video_paths = []
    for currentFile in currentDirectory.glob(currentPattern):
        print(currentFile)
        minute = int(currentFile.stem[-11:-9])
        # minute = int(currentFile.stem[-4:-2])
        if minute % 5 == 0:
            video_paths.append(currentFile)
    return video_paths


def run_computations(config):
    excel_path = config['excel_path']
    video_path = config['video_path']
    start_date = config['start_date']
    end_date = config['end_date']
    start_time = config['start_time']
    end_time = config['end_time']

    df = pd.read_excel(excel_path,
                       sheet_name='Configuration 1',
                       header=2, usecols=['Timestamps', ' m/s Wind Speed', ' m/s Gust Speed'],
                       index_col=0).dropna().loc[start_date:end_date].between_time(start_time, end_time)

    paths = get_paths(video_path=video_path)
    paths = paths[84:]  # TODO generalize, currently manual way to start at correct time!
    print(paths)

    vals = []
    vals_cut = []
    for indx, path in enumerate(paths):

        val, val_cut = get_stats(video_path=os.path.join(video_path, path.name),
                                 resize_scale=0.5, levels=80, winsize=4,
                                 show_rgb_motion=False, show_gray_motion=False, num_frames=80)
        print(path, df.iloc[indx, :], val, val_cut)
        vals.append(val)
        vals_cut.append(val_cut)
        if len(vals) == len(df):
            break
    print(vals)
    print(vals_cut)
    cv.destroyAllWindows()
    df['Mean Optical Flow'] = vals

    df['Wind Speed [m/s]'] = df[' m/s Wind Speed']

    return df


def result_visualizer(df):
    corr_wind_flow = df.corr().loc['Mean Optical Flow', 'Wind Speed [m/s]']
    print("corr wind flow:", corr_wind_flow)

    df.loc[:, ['Mean Optical Flow', 'Wind Speed [m/s]']].plot.line(subplots=True)
    plt.suptitle("Mean Optical Flow  & Wind Speed over Time")
    plt.xlabel("")
    plt.show()

    plt.xlim(0, 5)
    plt.ylim(0, 7)
    sns.regplot(x='Wind Speed [m/s]', y='Mean Optical Flow', data=df, truncate=False)
    plt.title('Mean Optical Flow vs Wind Speed')
    plt.show()


def main():
    # config = {'excel_path': 'z6-04349(z6-04349)-1611065058.xlsx',
    #           'video_path': './OneDrive_1_19-01-2021',
    #           'start_date': '2020-10-04',
    #           'end_date': '2020-10-04',
    #           'start_time': '10:05:00',
    #           'end_time': '11:00:00'}
    # config = {'excel_path': 'z6-04369(z6-04369)-1611067237.xlsx',
    #           'video_path': './181220-EEBED',
    #           'start_date': '2020-12-18',
    #           'end_date': '2020-12-18',
    #           'start_time': '09:10:00',
    #           'end_time': '11:49:00'}
    config = {'excel_path': 'z6-04349(z6-04349)-1610875144.xlsx',
              'video_path': './OneDrive_2_17-01-2021',
              'start_date': '2020-12-29',
              'end_date': '2020-12-29',
              'start_time': '14:00:00',  # 07:00
              'end_time': '17:20:00'}  # 17:45
    df = run_computations(config)
    result_visualizer(df)
    df_shift = df.copy()
    df_shift['Mean Optical Flow'] = df_shift['Mean Optical Flow'].shift(1)
    df_shift.dropna(inplace=True)
    result_visualizer(df_shift)

    pass
    pass


if __name__ == '__main__':
    main()
