import os

from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

def fit_predict_2df(train_path, test_path):
    df_train = pd.read_csv(os.path.join('results', train_path, "df_shift.csv"))

    reg = LinearRegression().fit(df_train.loc[:, 'Mean Optical Flow'].values.reshape(-1, 1),
                                 df_train.loc[:, 'Wind Speed [m/s]'].values.reshape(-1, 1))

    df_test = pd.read_csv(os.path.join('results', test_path, "df_shift.csv")).sample(20)

    res = reg.score(df_test.loc[:, 'Mean Optical Flow'].values.reshape(-1, 1),
                    df_test.loc[:, 'Wind Speed [m/s]'].values.reshape(-1, 1))
    print(res)

    plt.plot(df_train.loc[:, 'Mean Optical Flow'].values.reshape(-1, 1),
             df_train.loc[:, 'Wind Speed [m/s]'].values.reshape(-1, 1), 'x', label='Train')
    plt.plot(df_test.loc[:, 'Mean Optical Flow'].values.reshape(-1, 1),
             df_test.loc[:, 'Wind Speed [m/s]'].values.reshape(-1, 1), 'o', label='Test')
    plt.ylim(0, 7)
    plt.xlim(0, 7)
    plt.plot(np.arange(0, 8), reg.coef_.squeeze() * np.arange(0, 8) + reg.intercept_.squeeze())
    plt.xlabel('Mean Optical Flow')
    plt.ylabel('Wind Speed [m/s]')
    plt.legend()
    plt.show()

    y_pred_train = reg.predict(df_train.loc[:, 'Mean Optical Flow'].values.reshape(-1, 1))
    y_pred_test = reg.predict(df_test.loc[:, 'Mean Optical Flow'].values.reshape(-1, 1))
    # plt.plot(df_train.loc[:, 'Wind Speed [m/s]'].values.reshape(-1, 1), y_pred_train, 'o', label='Train')
    plt.plot(df_test.loc[:, 'Wind Speed [m/s]'].values.reshape(-1, 1), y_pred_test, 'x', label='Test')
    plt.plot(np.arange(0, 6), np.arange(0, 6))
    plt.legend()
    plt.xlabel('True Wind Speed')
    plt.ylabel('Predicted Wind Speed')
    plt.ylim(0, 4.5)
    plt.xlim(0, 4.5)

    plt.show()


def fit_predict(res_path):
    df_train = pd.read_csv(os.path.join('results', res_path, "df_shift.csv"))
    X_train, X_test, y_train, y_test = train_test_split(df_train.loc[:, 'Mean Optical Flow'].values.reshape(-1, 1),
                                                        df_train.loc[:, 'Wind Speed [m/s]'].values.reshape(-1, 1),
                                                        test_size=0.33, random_state=42)
    reg = LinearRegression().fit(X_train, y_train)
    # print(pearsonr(X_train.squeeze(), y_train.squeeze()))
    print(reg.score(X_train, y_train))
    print(reg.score(X_test, y_test))

    plt.plot(X_train, y_train, 'x', label='Train')
    plt.plot(X_test, y_test, 'o', label='Test')
    plt.ylim(0, 5)
    plt.xlim(0, 7)
    plt.plot(np.arange(0, 8), reg.coef_.squeeze() * np.arange(0, 8) + reg.intercept_.squeeze())
    plt.xlabel('Mean Optical Flow')
    plt.ylabel('Wind Speed [m/s]')
    plt.legend()
    plt.show()

    y_pred_train = reg.predict(X_train)
    y_pred_test = reg.predict(X_test)
    # plt.plot(y_train, y_pred_train, 'o', label='Train')
    plt.plot(y_test, y_pred_test, 'x', label='Validation')
    plt.plot(np.arange(0, 6), np.arange(0, 6))
    plt.legend()
    plt.xlabel('True Wind Speed [m/s]')
    plt.ylabel('Predicted Wind Speed [m/s]')
    plt.ylim(0, 4.5)
    plt.xlim(0, 4.5)
    plt.savefig("res1.svg")
    plt.show()


if __name__ == '__main__':
    # fit_predict(train_path='09022021_181047', test_path='09022021_183552')
    fit_predict(res_path='09022021_181047')
