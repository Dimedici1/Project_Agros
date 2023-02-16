"""
Description of script.
"""

import os
import pandas as pd


class Agros:
    """
    Some general description.

    Attributes
    ----------
    file_url: str
        Permalink of the file to be imported

    Methods
    --------
    import_file()
        Imports the file as pandas dataframe
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


FILE_URL = "https://github.com/owid/owid-datasets/blob/"\
            "693acdec5821af0a1b73523905d2a6ccefd6d509/datasets/"\
            "Agricultural%20total%20factor%20productivity%20(USDA)/"\
            "Agricultural%20total%20factor%20productivity%20(USDA).csv?raw=true"

agros = Agros(FILE_URL)
df = agros.import_file()
print(df.head())
