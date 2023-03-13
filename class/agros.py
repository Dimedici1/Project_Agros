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
from typing import List, Union
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pmdarima import auto_arima
import geopandas as gpd

warnings.filterwarnings('ignore')


class Agros:
    """
    A class that represents agricultural data and provides various methods for data analysis.

    Attributes
    ----------
    file_url : str
        A string representing the URL or path to the agricultural data file.

    agri_df : str
        A dataframe with data from the agricultural data file.

    merge_dict : dict()
        A dictionary to adapt the country names in the geo_data dataset.

    Methods
    -------
    import_file():
        Checks if the downloads folder exists and if agricultural_total_factor_productivity.csv
        is in the downloads folder. If not, the downloads folder is created and/or the the
        agricultural_total_factor_productivity.csv is downloaded from the web. If both the
        folder and the file exist, the file is loaded from the downloads folder and returned as
        a pandas dataframe.
        Furthermore, it downloads geographical information and stores it as a geopandas.
        geodataframe.GeoDataFrame.

    country_list():
        Creates a list of all unique countries/regions available in the dataset.

    corr_quantity():
        Calculates the correlation matrix for quantity-related columns of the
        agricultural dataframe and returns a heatmap plot using seaborn.

    area_chart():
        Plots an area chart of the distinct "_output_" columns. If a single country or a list
        of countries is specified, the method will plot area charts for each country.
        If no country, an empty list or "World" is specified, the method will plot the sum
        of the distinct outputs for all countries.
        If normalize is True, the output will be normalized in relative terms
        (output will always be 100% for each year).

    total_output():
        Plots the total of the distinct "_output_" columns per year for each country that
        is passed. Uses a line chart to make the total output comparable.

    gapminder():
        Plots a scatter plot to demonstrate the relationship between fertilizer and
        irrigation quantity on output for a specific year.

    choropleth():
        Plots a coropleth map that shows the tfp for every country in a specific year.

    predictor():
        Predicts the TFP (Total Factor Productivity) for up to three countries using
        ARIMA forecasting.
    """

    def __init__(self, file_url: str):
        self.file_url = file_url
        self.agri_df = None
        self.merge_dict = {
            "United States of America": "United States",
            "W. Sahara": "Sub-Saharan Africa",
            "Dem. Rep. Congo": "Democratic Republic of Congo",
            "Dominican Rep.": "Dominican Republic",
            "Timor-Leste": "Timor",
            "Central African Rep.": "Central African Republic"
        }

    def import_file(self) -> tuple:
        """
        Checks if the downloads folder exists and if agricultural_total_factor_productivity.csv
        is in the downloads folder. If not, the downloads folder is created and/or the the
        agricultural_total_factor_productivity.csv is downloaded from the web. If both the
        folder and the file exist, the file is loaded from the downloads folder and returned as
        a pandas dataframe.
        Furthermore, it downloads geographical information and stores it as a geopandas.
        geodataframe.GeoDataFrame.

        Returns
        ---------------
        tuple containing:

            agri_df: pandas dataframe
                A table with information from
                agricultural_total_factor_productivity.csv

            geo_data: A geopandas.geodataframe.GeoDataFrame file.
        """
        # Check if downloads folder exists
        downloads_dir = '../downloads'
        check_downloads_dir = os.path.isdir(downloads_dir)

        # If folder doesn't exist, create it.
        if not check_downloads_dir:
            os.makedirs(downloads_dir)

        file_path = os.path.join(
            "../downloads/agricultural_total_factor_productivity.csv"
        )
        # If file doesn't exist, create it.
        if not os.path.isfile(file_path):
            file_df = pd.read_csv(self.file_url, index_col=0)
            file_df.to_csv(file_path)

        self.agri_df = pd.read_csv(file_path, index_col=0)

        # Download geo data
        geo_data = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

        return self.agri_df, geo_data

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
                      "West Africa", "Western Europe", "World", "West Asia","Asia",
                      "North America", "Caribbean"]
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

        axes.set_title("Correlation Matrix of Quantity-Related Columns",
                       fontdict={'size':16, 'weight':'bold'}, pad=20)
        plt.xticks(rotation=45, ha="right")

        # Add an annotation
        axes.annotate("Source: International Agricultural Productivity,"
                      " USDA Economic Research Service, 2022",
                      xy=(0.5, -0.31), xycoords="axes fraction", fontsize=9,
                      ha='center', va='center', annotation_clip=False,
                      xytext=(0, 20), textcoords='offset points',
                      bbox={'boxstyle':'round,pad=0.5', 'fc':'white', 'alpha':0.9})

        return plt.show(corr_heatmap)

    def area_chart(self, country: Union[str, None, List[str]] = None, normalize: bool = False
                  ) -> plt.Axes:
        """
        Plots an area chart of the distinct "_output_" columns. If a single country or a list of
        countries is specified, the method will plot area charts for each country.
        If no country, an empty list or "World" is specified, the method will plot the sum of the
        distinct outputs for all countries.
        If normalize is True, the output will be normalized in relative terms
        (output will always be 100% for each year).

        Parameters
        ----------
        country : str, list of str or None, optional
            A string representing the name of the country to be plotted, or a list of strings
            representing the names of the countries to be plotted.
            If None, an empty list or "World", the chart will show the sum of the distinct outputs
            for all countries.

        normalize : bool, optional
            A boolean value indicating whether or not the output should be normalized.

        Returns
        -------
        axes: matplotlib.Axes
            The area chart as a Matplotlib axes object.

        Raises
        ------
        ValueError
            If any of the specified countries is not a country of the dataset.

        TypeError
            If normalize parameter is not a boolean value (True or False).

        TypeError
            If input for the country parameter is not a string or a list of strings.
        """
        if not isinstance(normalize, bool):
            raise TypeError("normalize parameter must be a boolean value (True or False)")

        # Define a color scheme for the areas
        colors = ["#43B02A", "#F9A71C", "#00A3E0"]

        # If no country or "World" is specified, plot the sum of the outputs for all countries
        if country in [None, [], "World"]:
            countries = self.country_list()
            data = self.agri_df.loc[countries].copy().groupby("Year").sum()
            title = "Global Agricultural Output"
            # Create a subset from original dataset with columns containing '_output_'
            data = data.pivot_table(
                index="Year", values=[col for col in data.columns if "_output_" in col]
            )
            data.rename(columns = {"animal_output_quantity": "Animal", "crop_output_quantity":
                                   "Crop", "fish_output_quantity": "Fish"}, inplace = True)

            # Normalize the data if necessary
            if normalize:
                data = data.div(data.sum(axis=1), axis=0)

            # Plot the area chart
            axes = data.plot.area(stacked=True, color=colors, alpha=0.8, linewidth=0)
            axes.set_xlabel("Year", fontsize=12, labelpad=10)
            axes.set_ylabel("Output (1000$)", fontsize=12, labelpad=10)
            axes.tick_params(axis='both', labelsize=10)
            axes.grid(axis='y', alpha=0.3)
            axes.set_title(title, fontdict={'size':16, 'weight':'bold'}, pad=20)

            # Add a legend
            handles, labels = axes.get_legend_handles_labels()
            axes.legend(handles, labels, fontsize=12, loc='upper left', bbox_to_anchor=(1.05, 1))

            # Add an annotation
            axes.annotate("Source: International Agricultural Productivity,"
                          " USDA Economic Research Service, 2022",
                          xy=(0.5, -0.31), xycoords="axes fraction", fontsize=9,
                          ha='center', va='center', annotation_clip=False,
                          xytext=(0, 20), textcoords='offset points',
                          bbox={'boxstyle':'round,pad=0.5', 'fc':'white', 'alpha':0.9})

        else:
            # Checks if input is a string and converts it to a list
            if isinstance(country, str):
                countries = [country]
            # Checks if input is a list
            elif isinstance(country, list):
                countries = country
            # Checks if input is a list (Or a string that was converted to a list)
            else:
                raise TypeError(f"{country} needs to be a string or a list of strings")

            for cou in countries:
                # If no country or "World" is specified, plot sum of outputs for all countries
                if cou in (None, "World"):
                    data = self.agri_df.loc[self.country_list()].copy().groupby("Year").sum()
                    title = "Global Agricultural Output"
                # Check if the input is an actual country
                if cou not in self.country_list() and cou is not None and cou != "World":
                    raise ValueError(f"{cou} is not a country of the dataset")
                if cou in self.country_list():
                    # If a country is specified, filter the dataframe to include only that country
                    data = self.agri_df.loc[cou, :].copy()
                    title = f"Agricultural Output of {cou}"
                # Create a subset from original dataset with columns containing '_output_'
                data = data.pivot_table(
                    index="Year", values=[col for col in data.columns if "_output_" in col]
                )
                data.rename(columns = {"animal_output_quantity": "Animal", "crop_output_quantity":
                                       "Crop", "fish_output_quantity": "Fish"}, inplace = True)

                # Normalize the data if necessary
                if normalize:
                    data = data.div(data.sum(axis=1), axis=0)

                # Plot the area chart
                axes = data.plot.area(stacked=True, color=colors, alpha=0.8, linewidth=0)
                axes.set_xlabel("Year", fontsize=12, labelpad=10)
                axes.set_ylabel("Output (1000$)", fontsize=12, labelpad=10)
                axes.tick_params(axis='both', labelsize=10)
                axes.grid(axis='y', alpha=0.3)
                axes.set_title(title, fontdict={'size':16, 'weight':'bold'}, pad=20)

                # Add a legend
                handles, labels = axes.get_legend_handles_labels()
                axes.legend(handles, labels, fontsize=12, loc='upper left',
                            bbox_to_anchor=(1.05, 1))

                # Add an annotation
                axes.annotate("Source: International Agricultural Productivity,"
                              " USDA Economic Research Service, 2022",
                              xy=(0.5, -0.31), xycoords="axes fraction", fontsize=9,
                              ha='center', va='center', annotation_clip=False,
                              xytext=(0, 20), textcoords='offset points',
                              bbox={'boxstyle':'round,pad=0.5', 'fc':'white', 'alpha':0.9})

        return plt.show(axes)

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
        axes = data.plot.area(stacked=True, alpha=0.8, linewidth=0)
        axes.set_xlabel("Year", fontsize=12, labelpad=10)
        axes.set_ylabel("Total Output (1000$)", fontsize=12, labelpad=10)
        axes.tick_params(axis='both', labelsize=10)
        axes.grid(axis='y', alpha=0.3)
        axes.set_title("Total Output per Country", fontdict={'size':16, 'weight':'bold'}, pad=20)

        # Add a legend
        handles, labels = axes.get_legend_handles_labels()
        axes.legend(handles, labels, fontsize=12, loc='upper left',
                    bbox_to_anchor=(1.05, 1))

        # Add an annotation
        axes.annotate("Source: International Agricultural Productivity,"
                      " USDA Economic Research Service, 2022",
                      xy=(0.5, -0.31), xycoords="axes fraction", fontsize=9,
                      ha='center', va='center', annotation_clip=False,
                      xytext=(0, 20), textcoords='offset points',
                      bbox={'boxstyle':'round,pad=0.5', 'fc':'white', 'alpha':0.9})

        return plt.show(axes)

    def gapminder(self, year: int, logscale: bool = False) -> plt.Axes:
        """
        Plots a scatter plot to demonstrate the relationship between fertilizer and
        irrigation quantity on output for a specific year.

        Parameters
        ----------
        year: int
            An integer that determines the year that will be selected from the data.
        logscale: bool
            A boolean that determines if the axes should be a logscale or linear scale

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
            x="fertilizer_quantity",
            y="output_quantity",
            s=irrigation_quantity_scaled,
            alpha=0.5
        )

        # Change the axes to a logarithmic scale if log scale is True
        if logscale:
            axes.set_xscale('log')
            axes.set_yscale('log')

        # Define the three size categories
        min_size = round(irrigation_quantity_scaled.mean(), 1)
        max_size = round(irrigation_quantity_scaled.max(), 1)
        med_size = round((min_size + max_size) / 2, 1)

        # create a custom legend
        legend_elements = [
            plt.scatter([], [], s=max_size, alpha=0.5, facecolor='blue',
                        label=f'Large ({max_size})'),
            plt.scatter([], [], s=med_size, alpha=0.5, facecolor='blue',
                        label=f'Medium ({med_size})'),
            plt.scatter([], [], s=min_size, alpha=0.5, facecolor='blue',
                        label=f'Small ({min_size})')
        ]

        axes.legend(handles=legend_elements, loc='upper left',
                   title="The diameter of each\ndot shows the irrigation\n"
                    "quantity (1000 ha) in\na country.\n",
                   bbox_to_anchor=(1.02, 1), borderaxespad=0)

        axes.set_title("Effect of Fertilizer and Irrigation Quantity on Output in " + str(year),
                       fontdict={'size': 16, 'weight': 'bold'}, pad=20)
        axes.set_xlabel("Fertilizer Quantity (t)", fontsize=12, labelpad=10)
        axes.set_ylabel("Output Quantity ($1000)", fontsize=12, labelpad=10)
        axes.tick_params(axis='both', labelsize=10)
        axes.grid(axis='both', alpha=0.3)

        # Add an annotation
        axes.annotate("Source: International Agricultural Productivity,"
                      " USDA Economic Research Service, 2022",
                      xy=(0.5, -0.32), xycoords="axes fraction", fontsize=9,
                      ha='center', va='center', annotation_clip=False,
                      xytext=(0, 20), textcoords='offset points',
                      bbox={'boxstyle':'round,pad=0.5', 'fc':'white', 'alpha':0.9})

        return plt.show(axes)


    def choropleth(self, year: int = 2019) -> plt.Axes:
        """
        Plots a choropleth map that shows the tfp for every country in a specific year.

        Parameters
        ----------
        year: int
            An integer that determines the year that will be selected from the data.

        Returns
        -------
        axes: matplotlib.Axes
            The choropleth chart as a Matplotlib axes object.

        Raises
        ------
        ValueError
            If any of the specified countries does not exist in the dataset.

        ValueError: cannot convert float NaN to integer
            If a year that is not in the dataset is selected.
        """
        # Checks if input is a list (Or a string that was converted to a list)
        if not isinstance(year, int):
            raise TypeError(f"{year} needs to be an integer")

        # Get data and geo data and merge it
        data, geo_data = self.import_file()
        geo_data['name'] = geo_data['name'].replace(self.merge_dict).fillna(geo_data['name'])
        merged_data = pd.merge(geo_data, data, left_on="name", right_on="Entity", how="left")
        merged_data = merged_data[merged_data['Year'] == year]

        #Plot the Choropleth map
        axes = merged_data.plot(column = 'tfp', #Assign numerical data column
                      legend = True, #Decide to show legend or not
                      figsize = [20,10],\
                      legend_kwds = {'label': "TFP per Country"}) #Name the legend

        # Add a title and axis labels
        axes.set_title(f"Total Factor Productivity (TFP) per Country in {year}",
                       fontdict={'size':16, 'weight':'bold'}, pad=20)
        axes.set_xlabel("Longitude", fontsize=12, labelpad=10)
        axes.set_ylabel("Latitude", fontsize=12, labelpad=10)

        # Add an annotation
        axes.annotate("Sources:\n1) International Agricultural Productivity,"
                      " USDA Economic Research Service, 2022\n"
                      "2) Made with Natural Earth. Free vector and raster map"
                      " data @ naturalearthdata.com, 2022",
                      xy=(0.275, -0.32), xycoords="axes fraction", fontsize=9,
                      ha='left', va='center', annotation_clip=False,
                      xytext=(0, 20), textcoords='offset points',
                      bbox={'boxstyle':'round,pad=0.5', 'fc':'white', 'alpha':0.9})

        return plt.show(axes)

    def predictor(self, countries: List[str]) -> plt.Axes:
        """
        Predicts the TFP (Total Factor Productivity) for up to three countries using
        ARIMA forecasting.

        Parameters
        ----------
        countries: List[str]
            A list of up to three country names to be included in the TFP prediction.

        Returns
        -------
        axes: plt.Axes
            A matplotlib axes object containing the plot of TFP for the selected countries
            and their ARIMA forecast.

        Raises
        ------
        TypeError:
            If the number of valid country names selected is not between 1 and 3.
        """
        # Check if country is in dataset and append it to the country_plot list
        country_plot = []
        for country in countries:
            if country in self.country_list():
                country_plot.append(country)

        # Raise TypeError if list with valid country names is 0 or larger than 3
        if len(country_plot) == 0 or len(country_plot) > 3:
            raise TypeError(f'Please select up to three of the following countries \
                            {self.country_list()}')

        # Initialize plot
        color_index = 0
        fig, axes = plt.subplots()

        for country in country_plot:
            # Use TFP column for each country to find ARIMA prediction parameters using stepwise fit
            data = self.agri_df.loc[country].set_index('Year')[['tfp']]
            data.index = pd.to_datetime(data.index, format='%Y')
            stepwise_fit = auto_arima(data['tfp'],
                                      start_p=1,
                                      start_q=1,
                                      max_p=3,
                                      max_q=3,
                                      m=1,
                                      start_P=0,
                                      seasonal=False,
                                      d=None,
                                      D=1,
                                      # trace=True,
                                      error_action='ignore',
                                      suppress_warnings=True,
                                      stepwise=True)

            n_periods = 2050 - max(data.index).year

            # Create dataframe with TFP predictios
            prediction = pd.DataFrame(stepwise_fit.predict(n_periods=n_periods))
            prediction['Time'] = pd.date_range(start='2020-01-01', periods=n_periods, freq='YS')
            prediction.set_index('Time', inplace=True)

            # Plot line of each country in a different color (dashed lines for predicitions)
            colors = ['blue', 'green', 'red']
            axes.plot(data, color=colors[color_index], label=country)
            axes.plot(prediction, color=colors[color_index],
                      linestyle='dashed',
                      label=f'{country} (ARIMA Forecast)')

            color_index += 1

        # Finalize plot
        axes.set_title('Evolution of Total Factor Productivity (TFP) Over Time',
                       fontdict={'size': 14, 'weight': 'bold'}, pad=20)
        axes.set_xlabel("Year", fontsize=12, labelpad=10)
        axes.set_ylabel("Total Factor Productivity (TFP)", fontsize=12, labelpad=10)
        axes.tick_params(axis='both', labelsize=10)
        axes.grid(axis='y', alpha=0.3)

        # Add a legend
        handles, labels = axes.get_legend_handles_labels()
        axes.legend(handles, labels, fontsize=12, loc='upper left',
                    bbox_to_anchor=(1.05, 1))

        # Add an annotation
        axes.annotate("Source: International Agricultural Productivity,"
                      " USDA Economic Research Service, 2022",
                      xy=(0.5, -0.31), xycoords="axes fraction", fontsize=9,
                      ha='center', va='center', annotation_clip=False,
                      xytext=(0, 20), textcoords='offset points',
                      bbox={'boxstyle':'round,pad=0.5', 'fc':'white', 'alpha':0.9})

        return plt.show(fig, axes)
