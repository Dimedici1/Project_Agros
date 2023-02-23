"""
Description:
    This script defines the Agros class, which represents agricultural
    data and provides various methods for data analysis.

Requirements:
    The script imports the following libraries: os, numpy, pandas,
    seaborn, and matplotlib.pyplot.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union


class Agros:
    """
    A class that represents agricultural data and provides various methods for data analysis.

    Parameters
    ----------
    file_url : str
        A string representing the URL or path to the agricultural data file.

    Attributes
    ----------
    file_url : str
        A string representing the URL or path to the agricultural data file.

    agri_df : pandas dataframe
        A table with information from the agricultural data file.

    Methods
    -------
    import_file():
        Checks if the agricultural data file is in the downloads folder and,
        if not, downloads it from the web. If it is already present in the downloads
        folder, the file is loaded from there and returned as a pandas dataframe.

    country_list():
        Creates a list of all unique countries/regions available in the dataset.

    corr_quantity():
        Calculates the correlation matrix for quantity-related columns of the
        agricultural dataframe and returns a heatmap plot using seaborn.
    """

    def __init__(self, file_url: str) -> pd.DataFrame:
        self.file_url = file_url

    def import_file(self):
        """
        Checks if agricultural_total_factor_productivity.csv
        is in the downloads folder and, if not, downloads
        it from the web. If it is there, the file is loaded from the
        downloads folder and returned as a pandas dataframe.

        Parameters
        ---------------
        self.file_url: str
            File permalink

        Returns
        ---------------
        self.agri_df: pandas dataframe
            A table with information from
            agricultural_total_factor_productivity.csv
        """
        file_path = os.path.join(
            "../downloads/agricultural_total_factor_productivity.csv"
        )

        if not os.path.isfile(file_path):
            file_df = pd.read_csv(self.file_url, index_col=0)
            file_df.to_csv(file_path)

        self.agri_df = pd.read_csv(file_path, index_col=0)
        return self.agri_df

    def country_list(self):
        """
        Creates a list of all unique countries/regions available
        in the dataset.

        Parameters
        ---------------
        self.agri_df: pandas dataframe
            A table with information from
            agricultural_total_factor_productivity.csv

        Returns
        ---------------
        list_of_countries: list
            A list of all unique countries/regions present
            in the dataset.
        """
        list_of_countries = list(self.agri_df.index.unique())
        return list_of_countries

    def corr_quantity(self) -> sns.matrix.ClusterGrid:
        """
        Calculates the correlation matrix for quantity-related columns of the
        agricultural dataframe and returns a heatmap plot using seaborn.

        Parameters
        ---------------
        self.agri_df: pandas dataframe
            A table with information from
            agricultural_total_factor_productivity.csv

        Returns:
        ---------------
        corr_heatmap: sns.matrix.ClusterGrid
            A seaborn heatmap plot of the correlation matrix for quantity-related columns.
        """
        # Create subset of self.agri_df that includes only colums that end with '_quantity'
        quantity_df = self.agri_df.loc[
            :,
            [
                item.split("_")[-1] == "quantity"
                for item in list(agros.import_file().columns)
            ],
        ]
        corr_df = quantity_df.corr()

        # Generate mask for the upper triangle
        mask = np.zeros_like(corr_df, dtype=bool)
        mask[np.triu_indices_from(mask)] = True

        # Set up the matplotlib figure and Generate colormap
        f, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.color_palette("Blues", as_cmap=True)

        # Draw the heatmap with the mask
        corr_heatmap = sns.heatmap(
            corr_df,
            mask=mask,
            cmap=cmap,
            annot=True,
            vmax=1,
            vmin=0.5,
            linewidths=0.5,
        )

        ax.set_title("Correlation Matrix of Quantity-Related Columns")
        plt.xticks(rotation=45, ha="right")

        return corr_heatmap

    def area_chart(self, country: Union[str, None] = None, normalize: bool = False):
        """
        Plots an area chart of the distinct "_output_" columns. If a country is specified, the chart will
        show the output for that country only, otherwise the chart will show the sum of the distinct outputs for all
        countries. If normalize is True, the output will be normalized in relative terms (output will always
        be 100% for each year).

        Parameters
        ----------
        country : str, optional
            A string representing the name of the country to be plotted.
        normalize : bool, optional
            A boolean value indicating whether or not the output should be normalized.

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
            The area chart as a Matplotlib axes object.

        Raises
        ------
        ValueError
            If the specified country does not exist in the dataset.
        """
        if country is not None and country != "World":
            if country not in self.agri_df.index.unique():
                raise ValueError(f"{country} does not exist in the dataset")

            data = self.agri_df.loc[country, :].copy()
            title = f"Agricultural Output of {country}"
        else:
            data = self.agri_df.copy()
            title = "Global Agricultural Output"

        data = data.pivot_table(
            index="Year", values=[col for col in data.columns if "_output_" in col]
        )
        if normalize:
            data = data.div(data.sum(axis=1), axis=0)

        ax = data.plot.area(title=title, stacked=True)
        ax.set_xlabel("Year")
        ax.set_ylabel("Output")

        return ax

    def total_output(self, countries: Union[str, list] = list) -> plt.Axes:
        """
        Plots the total of the distinct "_output_" columns per year for each country that is passed. Uses a line chart
        to make the total output comparable

        Parameters
        ----------
        country : str, list
            A string representing the name of the country to be plotted or alternatively a list containing multiple
            strings of countries.

        Returns
        -------
        matplotlib.Axes
            The line chart as a Matplotlib axes object.

        Raises
        ------
        ValueError
            If the specified country does not exist in the dataset.
            :param countries:
        """
        # Checks if input is a string
        global data
        if type(countries) == str:
            # Checks if country passed is in the dataset. Raises a ValueError if it is not.
            if countries not in self.agri_df.index.unique():
                raise ValueError(f"{countries} does not exist in the dataset")

            # Creates a DataFrame with a Year column and a Total output column for the country
            else:
                data = self.agri_df.loc[countries, :].copy()
                data = data.pivot_table(
                    index="Year",
                    values=[col for col in data.columns if "_output_" in col],
                )
                data[countries] = data.sum(axis=1)
                data.drop(
                    [col for col in data.columns if "_output_" in col],
                    axis=1,
                    inplace=True,
                )

        # Checks if the input is a list
        elif type(countries) == list:
            data = pd.DataFrame()

            # Creates a DataFrame with a Year column and one column for the total output of each country
            for country in countries:
                if country not in self.agri_df.index.unique():
                    raise ValueError(f"{country} does not exist in the dataset")
                else:
                    temporary = self.agri_df.loc[country, :].copy()
                    temporary = temporary.pivot_table(
                        index="Year",
                        values=[col for col in temporary.columns if "_output_" in col],
                    )
                    data[country] = temporary.sum(axis=1)

        # Plot the resulting DataFrame
        ax = data.plot.area(title="Total Output per Country", stacked=True)
        ax.set_xlabel("Year")
        ax.set_ylabel("Total Output")

        return ax


FILE_URL = (
    "https://github.com/owid/owid-datasets/blob/"
    "693acdec5821af0a1b73523905d2a6ccefd6d509/datasets/"
    "Agricultural%20total%20factor%20productivity%20(USDA)/"
    "Agricultural%20total%20factor%20productivity%20(USDA).csv?raw=true"
)


# Create an instance 'agros' of the Agros class
agros = Agros(FILE_URL)

# Apply methods to the 'agros' instance and print the results
print(agros.import_file())
print(agros.country_list())
print(agros.corr_quantity())
print(agros.area_chart())
# Yemen as an example for a string input
print(agros.total_output("Yemen"))
# Uruguay and Uzbekistan as an example for a list input
print(agros.total_output(["Uruguay", "Uzbekistan"]))
