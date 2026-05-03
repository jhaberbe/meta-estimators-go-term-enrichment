import numpy as np
import pandas as pd
from scipy.stats import norm

class GeneOntologyDifferentialResults:

    def meta_estimates(deseq_results, balanced_matrix):
        # weight / se ^ 2
        weighted_precision = balanced_matrix.div(deseq_results["lfcSE"].pow(2), axis=0)

        # Balanced estimate of log2change, based on weight and precision, we prefer estimates which are precise (1/se^2),
        # includes both the precision of the coefficent estimate and the number of samples), and are expressed highly (weight). 
        # We incorporate all those changes, and then normalize by the total weighted precision.
        meta_beta = (deseq_results['log2FoldChange'].values @ weighted_precision) / weighted_precision.sum(axis=0)

        # Weighted precision estimates
        # sqrt(sum of weights^2 * SE^2)
        q = (balanced_matrix.pow(2) / balanced_matrix.pow(2).sum(axis=0))
        meta_se = (q.pow(2).T * deseq_results["lfcSE"].pow(2)) \
            .T \
            .sum(axis=0) \
            .apply(np.sqrt)

        # Compute two tailed p-values
        pvalues = (2 * norm.sf(np.abs(meta_beta/meta_se)))

        df = pd.DataFrame({
            # Number of terms
            "n_terms": balanced_matrix.gt(0).sum(axis=0),

            # Meta Estimates
            "metaLFC": meta_beta, 
            "metaSE": meta_se,
            "lower_ci": meta_beta - 1.96 * meta_se,
            "upper_ci": meta_beta + 1.96 * meta_se,

            # Weight from IPF
            "weight": balanced_matrix.sum(axis=0),

            # Stats
            "pval": pvalues,
            "padj": pvalues * meta_beta.shape[0],
            "log10p": -np.log10(pvalues),
            "log10padj": -np.log10(pvalues * meta_beta.shape[0]),

            # Useful information about distribution of information.
            # We would like to avoid terms where one very significant gene makes the entire term informative.
            "top1_contribution": balanced_matrix.div(balanced_matrix.sum(axis=0), axis=1).max(axis=0),
            "top5_contribution": (balanced_matrix.div(balanced_matrix.sum(axis=0), axis=1).apply(lambda col: col.nlargest(5).sum(), axis=0)),
            "N_eff": 1 / (balanced_matrix / balanced_matrix.sum(axis=0)).pow(2).sum(),
        })

        return df