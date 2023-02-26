#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 17:36:09 2023

Authors:
    Luca Carocci, Anton Badort, Florian Preiss, Lorenzo Schumann

Description:
    This script defines the Agros class, which represents agricultural
    data and provides various methods for data analysis.

Requirements:
    The script imports the following libraries: os, typing.Union, numpy, pandas,
    seaborn, and matplotlib.pyplot.
"""

import os
from typing import Union
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Agros:
    """
    A class that represents agricultural data and provides various methods for data analysis.

    Attributes
    ----------
    file_url : str
        A string representing the URL or path to the agricultural data file.

    agri_df : str
        A dataframe with data from the agricultural data file.

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

    area_chart():
        Plots an area chart of the distinct "_output_" columns. If a country is specified, the
        chart will show the output for that country only, otherwise the chart will show the sum
        of the distinct outputs for all countries. If normalize is True, the output will be
        normalized in relative terms (output will always be 100% for each year).

    total_output():
        Plots the total of the distinct "_output_" columns per year for each country that
        is passed. Uses a line chart to make the total output comparable.

    gapminder():
        Plots a scatter plot to demonstrate the relationship between fertilizer and
        irrigation quantity on output for a specific year.
    """

    def __init__(self, file_url: str):
        self.file_url = file_url
        self.agri_df = None

    def import_file(self) -> pd.DataFrame:
        """
        Checks if agricultural_total_factor_productivity.csv
        is in the downloads folder and, if not, downloads
        it from the web. If it is there, the file is loaded from the
        downloads folder and returned as a pandas dataframe.

        Returns
        ---------------
        agri_df: pandas dataframe
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

    def country_list(self) -> list:
        """
        Creates a list of all unique countries/regions available
        in the dataset.

        Returns
        ---------------
        list_of_countries: list
            A list of all unique countries/regions present
            in the dataset.
        """
        list_of_countries = list(self.agri_df.index.unique())
        # Filter all exceptions
        exceptions = ["Central Africa", "Central African Republic", "Central America",
                      "Central Asia", "Central Europe", "Developed Asia", "Developed countries",
                      "East Africa", "Eastern Europe", "Europe", "Former Soviet Union",
                      "High income", "Horn of Africa", "Latin America and the Caribbean",
                      "Least developed countries", "Low income", "Lower-middle income",
                      "North Africa", "Northeast Asia", "Northern Europe", "Oceania",
                      "Pacific", "Sahel", "South Asia", "Southeast Asia", "Southern Africa",
                      "Southern Europe", "Sub-Saharan Africa", "Upper-middle income",
                      "West Africa", "Western Europe", "World", "West Asia"]
        list_of_countries = [x for x in list_of_countries if x not in exceptions]
        return list_of_countries

    def corr_quantity(self) -> sns.matrix.ClusterGrid:
        """
        Calculates the correlation matrix for quantity-related columns of the
        agricultural dataframe and returns a heatmap plot using seaborn.

        Returns:
        ---------------
        corr_heatmap: sns.matrix.ClusterGrid
            A seaborn heatmap plot of the correlation matrix for quantity-related columns.
        """
        # Create subset of self.agri_df that includes only colums that end with '_quantity'
        quantity_df = self.agri_df.loc[:,\
                                       [item.split("_")[-1] == "quantity"\
                                        for item in list(self.agri_df.columns)]]
        corr_df = quantity_df.corr()

        # Generate mask for the upper triangle
        mask = np.zeros_like(corr_df, dtype=bool)
        mask[np.triu_indices_from(mask)] = True

        # Set up the matplotlib figure and Generate colormap
        _, axes = plt.subplots(figsize=(11, 9))
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

        axes.set_title("Correlation Matrix of Quantity-Related Columns")
        plt.xticks(rotation=45, ha="right")

        return corr_heatmap

    def area_chart(
        self, country: Union[str, None] = None, normalize: bool = False
    ) -> plt.Axes:
        """
        Plots an area chart of the distinct "_output_" columns. If a country is specified, the
        chart will show the output for that country only, otherwise the chart will show the sum
        of the distinct outputs for all countries. If normalize is True, the output will be
        normalized in relative terms (output will always be 100% for each year).

        Parameters
        ----------
        country : str, optional
            A string representing the name of the country to be plotted.

        normalize : bool, optional
            A boolean value indicating whether or not the output should be normalized.

        Returns
        -------
        axes: matplotlib.Axes
            The area chart as a Matplotlib axes object.

        Raises
        ------
        ValueError
            If the specified country does not exist in the dataset.
        """
        # Check if the specified country exists in the dataset
        if country is not None and country != "World":
            if country not in self.agri_df.index.unique():
                raise ValueError(f"{country} does not exist in the dataset")
            # If a country is specified, filter the dataframe to include only that country
            data = self.agri_df.loc[country, :].copy()
            title = f"Agricultural Output of {country}"
        else:
            data = self.agri_df.copy()
            title = "Global Agricultural Output"
        # Create a subset from original dataset with columns containing '_output_'
        data = data.pivot_table(
            index="Year", values=[col for col in data.columns if "_output_" in col]
        )
        # Normalize the data if necessary
        if normalize:
            data = data.div(data.sum(axis=1), axis=0)
        # Plot the area chart
        axes = data.plot.area(title=title, stacked=True)
        axes.set_xlabel("Year")
        axes.set_ylabel("Output")

        return axes

    def total_output(self, countries: Union[str, list] = None) -> plt.Axes:
        """
        Plots the total of the distinct "_output_" columns per year for each country that
        is passed. Uses a line chart to make the total output comparable.

        Parameters
        ----------
        countries : str, list[str]
            A string representing the name of the country to be plotted or alternatively a
            list containing multiple strings of countries.

        Returns
        -------
        axes: matplotlib.Axes
            The line chart as a Matplotlib axes object.

        Raises
        ------
        ValueError
            If the specified country does not exist in the dataset.
        TypeError
            If the Parameters are not a str or a list of str.
        """
        # Import data and create DataFrame to store the output
        data = pd.DataFrame()
        # Checks if input is a string and converts it to a list
        if isinstance(countries, str):
            countries = [countries]
        # Checks if input is a list (Or a string that was converted to a list)
        if not isinstance(countries, list):
            raise TypeError(f"{countries} needs to be a string or a list of strings")

        for country in countries:
            # Checks if every value in the list is a string
            if not isinstance(country, str):
                raise TypeError(f"{country} needs to be a string")
            # Checks if the country is in the list of countries
            if country not in self.country_list():
                raise ValueError(f"{country} does not exist in the dataset")
            # Creates DataFrame with a Year column and a column for the total output of each country
            temporary = self.agri_df.loc[country, :].copy()
            temporary = temporary.pivot_table(
                index="Year",
                values=[col for col in temporary.columns if "_output_" in col],
            )
            data[country] = temporary.sum(axis=1)
        # Plot the resulting DataFrame
        axes = data.plot.area(title="Total Output per Country", stacked=True)
        axes.set_xlabel("Year")
        axes.set_ylabel("Total Output")

        return axes

    def gapminder(self, year: int) -> plt.Axes:
        """
        Plots a scatter plot to demonstrate the relationship between fertilizer and
        irrigation quantity on output for a specific year.

        Parameters
        ----------
        year: int
            An integer that determines the year that will be selected from the data.

        Returns
        -------
        axes: matplotlib.Axes
            The scatter plot as a Matplotlib axes object.

        Raises
        ------
        TypeError
            If the parameter passed is not an integer.
        """
        # Raise TypeError if year is not an integer
        if not isinstance(year, int):
            raise TypeError(f"{year} does not exist in the dataset")

        # Only consider data from the year passed
        dataframe = self.agri_df[self.agri_df["Year"] == year]

        # And only consider countries that are not are not part of exceptions
        dataframe = dataframe[dataframe.index.isin(self.country_list())]

        # Adjust the size of Irrigation quantity for plotting
        irrigation_quantity_scaled = dataframe["irrigation_quantity"] / 1000

        # Create the scatter plot
        axes = dataframe.plot.scatter(
            title="Effect of Fertilizer and Irrigation Quantity on Output",
            x="fertilizer_quantity",
            y="output_quantity",
            s=irrigation_quantity_scaled,
            alpha=0.5
        )
        # Define the three size categories
        min_size = round(irrigation_quantity_scaled.mean(), 1)
        max_size = round(irrigation_quantity_scaled.max(), 1)
        med_size = round((min_size + max_size)/2, 1)
        # create a custom legend
        legend_elements = [
            plt.scatter([], [], s=max_size, alpha=0.5, facecolor='blue',
                        label=f'Large ({max_size})'),
            plt.scatter([], [], s=med_size, alpha=0.5, facecolor='blue',
                        label=f'Medium ({med_size})'),
            plt.scatter([], [], s=min_size, alpha=0.5, facecolor='blue',
                        label=f'Small ({min_size})')
        ]

        plt.legend(handles=legend_elements, loc='upper left',
                   title='Each dot shows the irrigation\nquantity in one country\n')

        return axes
