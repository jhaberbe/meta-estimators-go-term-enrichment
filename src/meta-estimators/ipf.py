import numpy as np
import pandas as pd
from tqdm import tqdm

# Worst name ever.
class ReweightingPreparer:

    @staticmethod
    def clean(go_terms, deseq_results):
        joint_index = go_terms.index.intersection(deseq_results.index)

        go_terms = go_terms.loc[joint_index]
        go_terms = go_terms.loc[:, go_terms.sum(axis=0).gt(0)]
        deseq_results = deseq_results.loc[joint_index]

        row_marginals = deseq_results["baseMean"]
        column_marginals = go_terms.sum(axis=0)

        return deseq_results, go_terms, row_marginals, column_marginals

class IterativeProportionalFitting:

    def get_balanced_matrix(
        self, 
        assignment_matrix: pd.DataFrame, 
        row_marginals: pd.Series, 
        column_marginals: pd.Series,
        max_iter = 1000,
        tolerance = 1e-3
    ):

        # FIXME: I feel like this is bad form generally, to strip off the indices.
        A = assignment_matrix.values
        u = row_marginals.values
        v = column_marginals.values

        delta = 1

        with tqdm(range(max_iter), desc="IPF Running") as pbar:
            for _ in pbar:
                # Rescale
                A_u1 = ((A * u[:, None]) / A.sum(axis=0))
                A_u2 = ((A_u1 * v) / A_u1.sum(axis=0))

                # Compute % change in Frobenius Norm
                delta = np.pow((A - A_u2), 2).sum() / np.pow(A, 2).sum()
                pbar.set_postfix({"% Change (Frob Norm)": delta})

                # Update matrix
                A = A_u2

                if delta < tolerance:
                    return pd.DataFrame(
                        A,
                        index=assignment_matrix.index,
                        columns=assignment_matrix.columns
                    )

        return pd.DataFrame(
            A,
            index=assignment_matrix.index,
            columns=assignment_matrix.columns
        )