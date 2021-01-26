# https://www.geeksforgeeks.org/python-opencv-dense-optical-flow/
import cv2 as cv
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import os
import pandas as pd

import seaborn as sns


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
            return return_val_full / (index-1), return_val_cut / (index-1)

        index += 1
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
    return return_val_full / index, return_val_cut / index


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


def main():
    config = {'excel_path': 'z6-04349(z6-04349)-1611065058.xlsx',
              'video_path': './OneDrive_1_19-01-2021',
              'start_date': '2020-10-04',
              'end_date': '2020-10-04',
              'start_time': '10:05:00',
              'end_time': '11:00:00'}
    config = {'excel_path': 'z6-04369(z6-04369)-1611067237.xlsx',
              'video_path': './181220-EEBED',
              'start_date': '2020-12-18',
              'end_date': '2020-12-18',
              'start_time': '09:10:00',
              'end_time': '11:49:00'}

    excel_path =config['excel_path']
    video_path = config['video_path']
    start_date = config['start_date']
    end_date = config['end_date']
    start_time = config['start_time']
    end_time = config['end_time']
    # excel_path = 'z6-04349(z6-04349)-1610875144.xlsx'

    # video_path = './OneDrive_2_17-01-2021'

    # start_time = '10:00:00'
    # end_time = '17:44:00'

    df = pd.read_excel(excel_path,
                       sheet_name='Configuration 1',
                       header=2, usecols=['Timestamps', ' m/s Wind Speed', ' m/s Gust Speed'],
                       index_col=0).dropna().loc[start_date:end_date].between_time(start_time, end_time)

    paths = get_paths(video_path=video_path)
    print(paths)
    print("Press 'q' to quit.")
    vals = []
    vals_cut = []
    for indx, path in enumerate(paths):
        print(path)
        val, val_cut = get_stats(video_path=os.path.join(video_path, path.name),
                                 resize_scale=0.5, levels=30, winsize=1,
                                 show_rgb_motion=True, show_gray_motion=True, num_frames=4)
        vals.append(val)
        vals_cut.append(val_cut)
        if len(vals) == len(df):
            break
    print(vals)
    print(vals_cut)
    cv.destroyAllWindows()
    # vals = [0.9378691647376545, 1.9199790219907407, 3.874236062885802, 4.545384066358023, 3.0734112172067896, 2.1477523148148143, 3.8319313271604942, 2.7133967978395064, 3.5877570891203705, 2.6628880690586425, 2.986032359182098, 1.806789930555556, 2.876749855324074, 3.9423329957561735, 2.4202729552469138, 3.5481972415123457, 2.4749802276234574, 3.630550491898147, 3.5712999131944443, 3.7602235725308653, 3.422565441743828, 3.8309076967592604, 4.302838686342592, 3.708245563271606, 4.411617139274692, 2.677127893518519, 1.5893221223674652, 2.2309327980324074, 2.8998942901234566, 2.342068672839506, 2.7404666280864194, 3.7857180266203705]
    # vals_cut = [238.475, 110.725, 150.575, 155.525, 140.1, 137.85, 159.075, 196.2, 150.325, 159.575, 187.25, 234.45, 215.375, 150.975, 193.125, 193.0, 209.0, 149.625, 134.875, 190.975, 169.875, 99.075, 124.65, 108.45, 122.9, 151.575, 59.0, 140.375, 152.3, 196.325, 190.2, 141.65]    # 40 frames, between_time('07:00:00', '17:44:00'), resize_scale=0.8, levels=1, winsize=4,
    # vals = [2.960183188356001, 3.573699951171875, 5.09983100420163, 3.638677714783468, 4.302899753900222, 4.525741972746671, 3.5154221334575135, 5.883589530285493, 3.7697931548695514, 4.631827008282697, 3.7480234593520927, 3.909556730293933, 2.992668264883535, 4.4249002904067805, 2.5893580683955437, 2.7361432110821764, 2.3512735249083723, 4.333080037434896, 2.155874803331163, 2.7971161830572435, 2.1252620555736397, 2.397628087173273, 2.1796150301709587, 2.5239340322989, 2.337913513183594, 2.497829766921055, 1.9511263435269577, 2.465129089355469, 1.5242280559775274, 1.9033770902657214, 3.38627818543234, 3.0567101372612844, 4.132598933467158, 2.8629363448531544, 4.384891820836951, 2.443188495400511, 4.368668977713879, 2.250446686921296, 3.7801239955572434, 3.5489375173309705, 3.318651621956729, 3.5363610538435575, 3.6651543888044955, 3.636023514359086, 2.90377244360653, 3.4321129504545227, 3.2571487615137924, 3.7359056072470573, 4.128651786144867, 4.461869454089506, 4.174685103804976, 3.9024512208538282, 3.1643345773955915, 4.371897643289448, 2.891650013864777, 4.543621675467786, 5.6561337694709675, 4.338562689887152, 5.288275798669314, 4.607771320107542, 6.415146627543885, 2.593514920458381, 4.676713015709394, 4.476211321795427, 4.910565505793064, 3.9823368590555064, 4.984802340283805, 4.605336243429301, 3.309140749330875, 3.404384492683149, 4.656200644410687, 4.905572366993329, 4.855786596822462, 4.295453483675733, 2.4895999861352243, 5.009061592007861, 4.167841781804589, 5.323430699477962, 3.76244465392313, 6.258507226023137, 5.681622766565393, 4.798616423430266, 5.417231712812258, 4.682975884164148, 3.4834492247781625, 3.887790971920815, 4.839866958429783, 2.958713145314911, 3.4810501098632805, 2.404289547013648, 4.172451744550541, 3.1248964662905094, 5.409016041696807, 2.601462978786893, 2.763088311089409, 5.102267192322531, 3.6051486168378672, 3.925499433352621, 2.7230779953944833, 4.304321345576535, 6.472115316508727, 2.9689541663652594, 3.7332855601369594, 1.7386712345076198, 2.5551819412796584, 4.003097797911844, 2.572625694745852, 0.9589193461853782, 2.4354880627290703, 2.1227587287808642, 2.987096791208526, 1.9831630753882137, 2.1409592239945026, 3.265512197989005, 2.1158420138888894, 2.7134005888008788, 2.338218707802855, 2.4112029652536644, 2.8345964596595303, 2.934379238552518, 1.689573707109616, 1.616340109742718, 2.1681472966700417, 1.4673855063356, 2.045680670090664, 1.310308649510513, 1.217309589150511, 1.9213488731855228, 2.23963798240379]
    df['Mean Optical Flow'] = vals  # / max(vals)
    # df['Mean Optical Flow Cut'] = vals_cut #/ max(vals_cut)
    # df['wind_normalized'] = df[' m/s Wind Speed']/df[' m/s Wind Speed'].max()
    # df['wind_normalized Gust'] = df[' m/s Gust Speed']/df[' m/s Gust Speed'].max()
    # df= df.drop([' m/s Wind Speed',' m/s Gust Speed'], axis=1)
    print(df.corr())
    # fix, ax = plt.subplots(2, sharex=True, figsize=(12, 6))
    # df.loc[:, ['video_val', 'video_val_cut']].plot.line(ax=ax[0])
    # df.loc[:, [' m/s Wind Speed', ' m/s Gust Speed']].plot.line(ax=ax[1])
    # plt.show()
    df['Wind Speed [m/s]'] = df[' m/s Wind Speed']
    df.loc[:, ['Mean Optical Flow', 'Wind Speed [m/s]']].plot.line(subplots=True)
    plt.suptitle("Mean Optical Flow  & Wind Speed over Time")
    plt.xlabel("")
    plt.show()
    # df.plot.scatter(' m/s Wind Speed', 'video_val')
    plt.xlim(0, 5)
    plt.ylim(0, 7)
    # plt.show()
    pass
    print(1)
    pass
    sns.set_theme(color_codes=True)
    sns.regplot(x='Wind Speed [m/s]', y='Mean Optical Flow', data=df, truncate=False)
    # plt.xlabel('Wind Speed [m/s]')
    # plt.ylabel('Mean Optical Flow')
    plt.title('Mean Optical Flow vs Wind Speed')
    plt.show()
    # while True:
    #
    #     display_motion(video_path="P201229_070800_070900.avi",
    #                    resize_scale=0.8, levels=30, winsize=10,
    #                    show_rgb_motion=True, show_gray_motion=True)

    """
    levels=12/20, winsize=4:  0.270269
    first 3 imgs : 0.281592, 0.306387

    """


if __name__ == '__main__':
    main()
