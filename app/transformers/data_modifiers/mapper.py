import pandas as pd
from abc import ABC, abstractmethod

from app.transformers.data_modifiers.namespace import (
    SkiStyle,
    SkierLevel,
    Slope,
    STYLE_COLUMN,
    SKIER_LEVEL_COLUMN,
    SLOPE_COLUMN,
)


class Mapper(ABC):
    """
    Abstract base class for mapping operations on dataframes.
    """

    @abstractmethod
    def map(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps the input dataframe and returns a transformed dataframe.

        Args:
            df: Input pandas DataFrame

        Returns:
            Transformed pandas DataFrame
        """
        pass


class StyleMapper(Mapper):
    """
    Mapper that adds STYLE, SKIER_LEVEL, and SLOPE columns to the dataframe
    with constant values for each row.
    """

    def __init__(self, style: SkiStyle, skier_level: SkierLevel, slope: Slope):
        """
        Initialize the StyleMapper with constant values.

        Args:
            style: Value for the STYLE column
            skier_level: Value for the SKIER_LEVEL column
            slope: Value for the SLOPE column
        """
        self.style = style
        self.skier_level = skier_level
        self.slope = slope

    def map(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds STYLE, SKIER_LEVEL, and SLOPE columns to the dataframe with
        constant values defined during initialization.

        Args:
            df: Input pandas DataFrame

        Returns:
            DataFrame with added columns
        """
        # Create a copy to avoid modifying the original dataframe
        result_df = df.copy()

        # Add the constant columns
        result_df[STYLE_COLUMN] = self.style
        result_df[SKIER_LEVEL_COLUMN] = self.skier_level
        result_df[SLOPE_COLUMN] = self.slope

        return result_df
