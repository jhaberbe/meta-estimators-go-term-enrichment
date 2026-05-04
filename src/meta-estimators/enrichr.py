import requests
import pandas as pd
from functools import reduce

class GeneOntologyLibraries:
    _base_url = "https://maayanlab.cloud/Enrichr/geneSetLibrary"

    def pull_library(self, library_name: str):
        params = {
            "mode": "text",
            "libraryName": library_name,
        }
        response = requests.get(self._base_url, params=params)
        response.raise_for_status()

        return {
            line.split("\t\t")[0]: line.split("\t\t")[1].rstrip("\n").split("\t")
            for line in response.text.strip().splitlines()
        }

    def pull_assignment_matrix(self, library_name: str):
        # Pull the library
        library = self.pull_library(library_name)

        # Fill the assignment matrix
        return pd.DataFrame({k: {f: 1 for f in v} for k, v in library.items()}).fillna(0)