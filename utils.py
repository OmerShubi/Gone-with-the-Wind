
import matplotlib.pyplot as plt
import logging
import seaborn as sns
import os

def create_and_configer_logger(log_name):
    """
    TODO
    Args:
        log_name ():
    Returns:
    """
    # set up logging to file
    logging.basicConfig(
        filename=log_name,
        level=logging.DEBUG,
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    )

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
    return logger


def result_visualizer(df, save_path, shift_str=''):
    """
    Creates, shows and saves two graphs,
        1. Mean Optical Flow  & Wind Speed over Time
        2. Mean Optical Flow vs Wind Speed

    :param df: dataframe, should have columns of atleast ['Mean Optical Flow', 'Wind Speed [m/s]' ]
    """
    logger = logging.getLogger()

    corr_wind_flow = df.corr().loc['Mean Optical Flow', 'Wind Speed [m/s]']
    logger.debug(f"corr wind flow {shift_str}: {corr_wind_flow}")


    df.loc[:, ['Mean Optical Flow', 'Wind Speed [m/s]']].plot.line(subplots=True)
    plt.suptitle("Mean Optical Flow  & Wind Speed over Time")
    plt.xlabel("")
    plt.savefig(os.path.join(save_path, f'flow_vs_wind_vs_time{shift_str}.png'))
    plt.show()

    plt.xlim(0, 5)
    plt.ylim(0, 7)
    sns.regplot(x='Wind Speed [m/s]', y='Mean Optical Flow', data=df, truncate=False)
    plt.title('Mean Optical Flow vs Wind Speed')
    plt.savefig(os.path.join(save_path, f'flow_vs_wind{shift_str}.png'))
    plt.show()



