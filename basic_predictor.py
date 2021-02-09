import os

from sklearn.linear_model import LinearRegression
import pandas as pd


def fit_predict(res_path):
    df = pd.read_csv(os.path.join('results', res_path, "df_shift.csv"))
    reg = LinearRegression().fit(df.loc[:, 'Wind Speed [m/s]'].values.reshape(-1, 1),
                                 df.loc[:, 'Mean Optical Flow'].values.reshape(-1, 1))
    print(reg.predict([[5]]))


if __name__ == '__main__':
    fit_predict(res_path='09022021_181047')
