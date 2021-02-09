import os

from sklearn.linear_model import LinearRegression
import pandas as pd


def fit_predict(train_path, test_path):
    df = pd.read_csv(os.path.join('results', train_path, "df_shift.csv"))
    reg = LinearRegression().fit(df.loc[:, 'Mean Optical Flow'].values.reshape(-1, 1),
                                 df.loc[:, 'Wind Speed [m/s]'].values.reshape(-1, 1))
    # print(reg.predict([[2]]))
    df_test = pd.read_csv(os.path.join('results', test_path, "df_shift.csv"))

    res = reg.score(df_test.loc[:, 'Mean Optical Flow'].values.reshape(-1, 1),
                    df_test.loc[:, 'Wind Speed [m/s]'].values.reshape(-1, 1))
    print(res)


if __name__ == '__main__':
    fit_predict(train_path='09022021_181047', test_path='09022021_181047')
