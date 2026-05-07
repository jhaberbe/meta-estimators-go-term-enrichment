import numpy as np
import pandas as pd
from scipy import sparse
import numpy as np
from tqdm.auto import tqdm

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

    def sinkhorn_binary_support(
        self,
        row_marginals,
        column_marginals,
        A,
        prior="row_degree",
        max_iter=10_000,
        tol=1e-4,
        eps=1e-30,
    ):
        r = np.asarray(row_marginals, dtype=float)
        c = np.asarray(column_marginals, dtype=float)

        if np.any(r < 0) or np.any(c < 0):
            raise ValueError("Marginals must be nonnegative.")

        r_total = r.sum()
        c_total = c.sum()

        if not np.isclose(r_total, c_total, rtol=1e-8, atol=1e-12):
            c = c * (r_total / c_total)

        # Bookkeeping
        new_index = A.index
        new_columns = A.columns

        A = sparse.csr_matrix(A, dtype=float)
        A.data[:] = 1.0

        n_rows, n_cols = A.shape

        if r.shape[0] != n_rows:
            raise ValueError("row_marginals length does not match A.shape[0].")

        if c.shape[0] != n_cols:
            raise ValueError("column_marginals length does not match A.shape[1].")

        row_degree = np.asarray(A.sum(axis=1)).ravel()
        col_degree = np.asarray(A.sum(axis=0)).ravel()

        if np.any((r > 0) & (row_degree == 0)):
            raise ValueError("Some positive-mass rows have no allowed outcomes.")

        if np.any((c > 0) & (col_degree == 0)):
            raise ValueError("Some positive-mass columns have no allowed features.")

        A = A.tocoo()
        rows = A.row
        cols = A.col

        if prior == "support":
            data = np.ones_like(A.data, dtype=float)

        elif prior == "row_degree":
            data = 1.0 / np.maximum(row_degree[rows], eps)

        elif prior == "degree_corrected":
            data = 1.0 / np.sqrt(
                np.maximum(row_degree[rows], eps) * np.maximum(col_degree[cols], eps)
            )

        else:
            raise ValueError(
                "prior must be one of {'support', 'row_degree', 'degree_corrected'}."
            )

        K = sparse.coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols)).tocsr()

        u = np.ones(n_rows)
        v = np.ones(n_cols)

        converged = False

        for iteration in tqdm(range(max_iter)):
            Kv = K @ v
            u = np.divide(r, Kv, out=np.zeros_like(r), where=Kv > eps)

            Ktu = K.T @ u
            v = np.divide(c, Ktu, out=np.zeros_like(c), where=Ktu > eps)

            if iteration % 25 == 0 or iteration == max_iter - 1:
                row_sums = u * (K @ v)
                col_sums = v * (K.T @ u)

                row_err = np.max(np.abs(row_sums - r) / np.maximum(1.0, r))
                col_err = np.max(np.abs(col_sums - c) / np.maximum(1.0, c))

                if max(row_err, col_err) < tol:
                    converged = True
                    break

        X = K.tocoo()
        X_data = X.data * u[X.row] * v[X.col]
        X = sparse.coo_matrix((X_data, (X.row, X.col)), shape=K.shape).tocsr()

        balanced_matrix = pd.DataFrame(
            X.todense(),
            index=new_index,
            columns=new_columns
        )

        return balanced_matrix
    
    def degree_corrected_doubly_stochastic(
        self,
        A,
        max_iter=10_000,
        tol=1e-3,
        min_percent_change=1e-3,
        patience=25,
        eps=1e-12,
        verbose=True,
    ):
        """
        Convert a nonnegative binary/weighted matrix into a doubly stochastic matrix
        using Sinkhorn-Knopp scaling.

        Stops when:
        1. row/column sums are within `tol` of 1, or
        2. convergence stalls, measured by percent change in error.
        """

        A = np.asarray(A, dtype=float)

        if np.any(A < 0):
            raise ValueError("A must be nonnegative.")

        if np.any(A.sum(axis=1) == 0):
            raise ValueError(
                "A has at least one all-zero row, so exact row stochasticity is impossible."
            )

        if np.any(A.sum(axis=0) == 0):
            raise ValueError(
                "A has at least one all-zero column, so exact column stochasticity is impossible."
            )

        n, m = A.shape
        r = np.ones(n)
        c = np.ones(m)

        prev_error = np.inf
        stalled_iters = 0
        converged = False
        stopped_due_to_stall = False

        iterator = range(1, max_iter + 1)
        if verbose:
            iterator = tqdm(iterator, desc="Sinkhorn scaling")

        for iteration in iterator:
            row_sums = A @ c
            r = 1.0 / np.maximum(row_sums, eps)

            col_sums = A.T @ r
            c = 1.0 / np.maximum(col_sums, eps)

            P = r[:, None] * A * c[None, :]

            row_error = np.max(np.abs(P.sum(axis=1) - 1))
            col_error = np.max(np.abs(P.sum(axis=0) - 1))
            error = max(row_error, col_error)

            if np.isfinite(prev_error) and prev_error > 0:
                percent_change = 100 * (prev_error - error) / prev_error
            else:
                percent_change = np.nan

            if verbose:
                iterator.set_postfix(
                    {
                        "err": f"{error:.2e}",
                        "row": f"{row_error:.2e}",
                        "col": f"{col_error:.2e}",
                        "%chg": f"{percent_change:.2e}",
                        "stall": stalled_iters,
                    }
                )

            if error < tol:
                converged = True
                break

            if np.isfinite(percent_change):
                if percent_change < min_percent_change:
                    stalled_iters += 1
                else:
                    stalled_iters = 0

            if stalled_iters >= patience:
                stopped_due_to_stall = True
                break

            prev_error = error

        P = r[:, None] * A * c[None, :]

        info = {
            "iterations": iteration,
            "final_error": error,
            "final_row_error": row_error,
            "final_col_error": col_error,
            "converged": converged,
            "stopped_due_to_stall": stopped_due_to_stall,
            "stalled_iters": stalled_iters,
        }

        return P, r, c, info