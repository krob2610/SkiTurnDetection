from typing import List

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# dataset needs to be unpack
class Smoother:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def smooth_data_rolling_mean(
        self, window_size: int, columns: List[str]
    ) -> pd.DataFrame:
        df_smooth = self.data.copy()
        df_smooth[columns] = df_smooth[columns].rolling(window=window_size).mean()
        return df_smooth

    def smooth_data_exponential(self, span: int, columns: List[str]) -> pd.DataFrame:
        df_smooth = self.data.copy()
        df_smooth[columns] = df_smooth[columns].ewm(span=span).mean()
        return df_smooth

    @DeprecationWarning
    def smooth_data_holt_winters(
        self, columns: List[str], seasonal_periods: int
    ) -> pd.DataFrame:
        df_smooth = self.data.copy()
        for column in columns:
            model = ExponentialSmoothing(
                df_smooth[column], seasonal_periods=seasonal_periods
            )
            model_fit = model.fit()
            df_smooth[column] = model_fit.fittedvalues
        return df_smooth

    @DeprecationWarning
    def smooth_data_savgol(
        self, window_length: int, polyorder: int, columns: List[str]
    ) -> pd.DataFrame:
        df_smooth = self.data.copy()
        for column in columns:
            df_smooth[column] = savgol_filter(
                df_smooth[column], window_length, polyorder
            )
        return df_smooth

    def smooth_data_backward_ma(
        self, window_size: int, columns: List[str]
    ) -> pd.DataFrame:
        df_smooth = self.data.copy()
        for column in columns:
            smoothed = (
                df_smooth[column]
                .rolling(window=window_size)
                .mean()
                .shift(-(window_size - 1))
            )
            df_smooth[column] = smoothed.fillna(df_smooth[column])
        return df_smooth

    def smooth_data_centered_ma(
        self, window_size: int, columns: List[str]
    ) -> pd.DataFrame:
        df_smooth = self.data.copy()
        for column in columns:
            # Użycie centralnej średniej ruchomej z opcją center=True
            smoothed = df_smooth[column].rolling(window=window_size, center=True).mean()
            # Uzupełnianie brakujących wartości oryginalnymi danymi, aby uniknąć NaN na brzegach
            df_smooth[column] = smoothed.fillna(df_smooth[column])
        return df_smooth
