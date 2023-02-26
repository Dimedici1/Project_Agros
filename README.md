# Project Agros: Analyzing Agricultural Total Factor Productivity
Agros is a Python package for analyzing and visualizing data on agricultural total factor productivity. It was developed by Group 2 as part of a project on agricultural economics.

## Description
Agros is a Python-based project that focuses on analyzing Agricultural Total Factor Productivity (TFP). The aim of the project is to provide a tool that can measure the efficiency of agricultural production by evaluating the ratio of total output to total input in a given production process. The resulting analysis can help policymakers and farmers understand the factors that affect agricultural productivity and identify areas for improvement.

## Installation
To use Agros, you need to have Python 3.x installed on your system. You can then install the package using pip:

```
pip install agros
```

Alternatively, you can clone this repository and install the package from source:

```
git clone https://gitlab.com/florianpreiss/group_02.git
cd agros
pip install .
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

#### __init__(self, file_url: str) -> None
Initializes the Agros class with a file URL to download agricultural data.

#### import_file(self) -> pd.DataFrame
Downloads the agricultural data file if it is not in the downloads folder and returns it as a Pandas DataFrame. If the file is already present in the downloads folder, it will be loaded from there.

#### country_list(self) -> list
Returns a list of all unique countries/regions available in the dataset.

#### corr_quantity(self) -> sns.matrix.ClusterGrid
Calculates the correlation matrix for quantity-related columns of the agricultural DataFrame and returns a heatmap plot using Seaborn.

#### area_chart(self, country: Union[str, None] = None, normalize: bool = False) -> plt.Axes
Plots an area chart of the distinct "output" columns. If a country is specified, the chart will show the output for that country only. Otherwise, the chart will show the sum of the distinct outputs for all countries. If normalize is True, the output will be normalized in relative terms (output will always be 100% for each year).

#### total_output(self, countries: list) -> plt.Axes
Plots the total of the distinct "output" columns per year for each country that is passed. Uses a line chart to make the total output comparable.

#### gapminder(self, year: int) -> plt.Axes
Plots a scatter plot to demonstrate the relationship between fertilizer and irrigation quantity on output for a specific year.

## Usage
Agros provides a Agros class with several methods for analyzing and visualizing data on agricultural total factor productivity. Here's an example of how to use it:

#### Import class
```
from agros import Agros
```

#### Create an instance 'agros' of the Agros class
```
agros = Agros(FILE_URL)
```

#### Load the data
```
agros.import_file()
```

#### Analyze the data
```
agros.corr_quantity()
agros.corr_quantity()
agros.area_chart()
agros.total_output()
agros.gapminder()
```

For more examples, please refer to the showcase notebook (showcase.ipynb) in this repository.

## Authors
Agros was developed by Group 2 as part of a project on agricultural economics. The authors are:

- Florian Preiss (@florianpreiss)
- Anton Badort (@ABdrt)
- Lorenzo Schumann (@Dimedici)
- Luca Carocci (@carocciluca)

## Acknowledgment
We would like to thank the USDA for providing the data used in this project.

## License
Agros is released under the GNU General Public License v3.0. See LICENSE for more information.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
