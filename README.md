# Project Agros: Analyzing Agricultural Total Factor Productivity
Agros is a Python package for analyzing and visualizing data on agricultural total factor productivity. It was developed by Group 2 as part of a project on agricultural economics.

## Description
Agros is a Python-based project designed to analyze agricultural total factor productivity (TFP). The project aims to provide a tool that measures the efficiency of agricultural production by evaluating the ratio of total output to total input in a given production process. This analysis can be helpful for policy makers and farmers in understanding the factors that influence agricultural productivity and identifying areas for improvement.

TFP is a significant measure of agricultural productivity, as it measures the amount of agricultural output produced from a combination of land, labor, capital, and physical resources used in agricultural production. If the total output grows faster than the total inputs, the total factor productivity (TFP) increases.

The dataset used in this project is provided by the U.S. Department of Agriculture (USDA) Economic Research Service. It contains annual indices of agricultural TFP for countries and regions of the world, from 1961 to 2019. More detailed information on the dataset and TFP can be found [here](https://www.ers.usda.gov/data-products/international-agricultural-productivity/).

**Variables in the Economic Research Service, USDA, International Agricultural Productivity dataset**

| Variable Name | Description | Unit |
| --- | --- | --- |
| Entity | Country/Territory name | |
| Year | Year | |
| TFP | agricultural TFP | |
| Output | total agricultural output | |
| inputs | total agricultural input | |
| ag_land_index | Index of total agricultural land input | Index |
| labor_index | Index of total agricultural labor input | Index |
| capital_index | Index of total agricultural capital input | Index |
| materials_index | Index of total agricultural materials input | Index |
| output_quantity | Quantity of total agricultural output | $1000 |
| crop_output_quantity | Quantity of total crop output | $1000 |
| animal_output_quantity | Quantity of total animal output | $1000 |
| fish_output_quantity | Quantity of total aquaculture output | $1000 |
| ag_land_quantity | Quantity of total agricultural land | 1000 hectares, rainfed-cropland-equivalents |
| labor_quantity | Quantity of total agricultural labor | 1000 persons economically active in agriculture |
| capital_quantity | Quantity of total agricultural capital stock | $million |
| machinery_quantity | Quantity of total agricultural machinery stock | 1000 metric horsepower (CV) |
| livestock_quantity | Quantity of total agricultural animal inventories | 1000 head of standard livestock units |
| fertilizer_quantity | Quantity of total agricultural fertilizer | Metric tons of inorganic N,P,K and organic N |
| animal_feed_quantity | Quantity of total agricultural feed | 106 Mcal of metabolizable energy |
| cropland_quantity | Quantity of total cropland | 1000 hectares |
| pasture_quantity | Quantity of total permanent pasture | 1000 hectares |
| irrigation_quantity | Quantity of total area equipped for irrigation | 1000 hectares |


## Installation
To use Agros, you need to have Python 3.x installed on your system. You can then download the source code [here](https://gitlab.com/florianpreiss/group_02/-/archive/main/group_02-main.zip) or you can clone this repository.

```
git clone https://gitlab.com/florianpreiss/group_02.git
```

## Requirements
The following libraries are required to run the Agros class:

- os
- typing.Union
- numpy
- pandas
- seaborn
- matplotlib.pyplot

## Methods
The Agros class provides the following methods:

**__init__(self, file_url: str) -> None**

Initializes the Agros class with a file URL to download agricultural data.

**import_file(self) -> pd.DataFrame**

Downloads the agricultural data file if it is not in the downloads folder and returns it as a Pandas DataFrame. If the file is already present in the downloads folder, it will be loaded from there.

**country_list(self) -> list**

Returns a list of all unique countries/regions available in the dataset.

**corr_quantity(self) -> sns.matrix.ClusterGrid**

Calculates the correlation matrix for quantity-related columns of the agricultural DataFrame and returns a heatmap plot using Seaborn.

**area_chart(self, country: Union[str, None] = None, normalize: bool = False) -> plt.Axes**

Plots an area chart of the distinct "output" columns. If a country is specified, the chart will show the output for that country only. Otherwise, the chart will show the sum of the distinct outputs for all countries. If normalize is True, the output will be normalized in relative terms (output will always be 100% for each year).

**total_output(self, countries: list) -> plt.Axes**

Plots the total of the distinct "output" columns per year for each country that is passed. Uses a line chart to make the total output comparable.

**gapminder(self, year: int) -> plt.Axes**

Plots a scatter plot to demonstrate the relationship between fertilizer and irrigation quantity on output for a specific year.

**predictor(self, countries: List[str]) -> plt.Axes**

Predicts the TFP (Total Factor Productivity) for up to three countries using ARIMA forecasting.

## Usage
Agros provides a Agros class with several methods for analyzing and visualizing data on agricultural total factor productivity. Here's an example of how to use it:

**Import class**

```
import sys
sys.path.append(sys.path[0] + "\\class")
from agros import Agros
```

**Create an instance 'agros' of the Agros class**

```
FILE_URL = (
    "https://github.com/owid/owid-datasets/blob/"
    "693acdec5821af0a1b73523905d2a6ccefd6d509/datasets/"
    "Agricultural%20total%20factor%20productivity%20(USDA)/"
    "Agricultural%20total%20factor%20productivity%20(USDA).csv?raw=true"
)
agros = Agros(FILE_URL)
```

**Load the data**

```
agros.import_file()
```

**Analyze the data**

```
agros.corr_quantity()
agros.corr_quantity()
agros.area_chart()
agros.total_output()
agros.gapminder()
agros.predictor()
```

For more examples, please refer to the showcase notebook (showcase.ipynb) in this repository.

## Authors
Agros was developed by Group 2 as part of a project on agricultural economics. The authors are:

- Florian Preiss, 54385, 54385@novasbe.pt (@florianpreiss)
- Anton Badort, 55358, 55358@novasbe.pt (@ABdrt)
- Lorenzo Schumann, 56178, 56178@novasbe.pt (@Dimedici)
- Luca Carocci, 53942, 53942@novasbe.pt (@carocciluca)

## Acknowledgment
We would like to thank the USDA for providing the data used in this project.

## License
Agros is released under the GNU General Public License v3.0. See LICENSE for more information.

## Project status
Agros is currently in active development, and new features and improvements are being added on an ongoing basis. While the current version of the project is functional and can provide valuable insights into agricultural total factor productivity, there are still some areas that require further development and optimization. We are committed to continuously improving and updating the project to ensure that it remains a useful tool for researchers and policymakers in the field of agriculture.
