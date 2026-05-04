import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 
# Base plot was made by myself, then chatgpt made it actually pretty.

class ForestPlotter:

    def __init__(self, deseq_results, balanced_matrix, gene_ontology_results):
        self.deseq_results = deseq_results
        self.balanced_matrix = balanced_matrix
        self.gene_ontology_results = gene_ontology_results

    def plot_term(
        self,
        term: str,
        cutoff=0.99,
        figsize=None,
        ci_multiplier=1.96,
        point_size=45,
        contribution_label=True,
        title=None,
    ):
        contributing_genes = self.balanced_matrix.index[
            self.balanced_matrix[term].gt(0)
        ]

        subset_df = self.deseq_results.loc[contributing_genes].copy()
        subset_weights = self.balanced_matrix.loc[contributing_genes, term]

        contribution = subset_weights / subset_weights.sum()
        meta_estimation = self.gene_ontology_results.loc[term]

        forest_df = pd.DataFrame({
            "gene": subset_df.index,
            "L2FC": subset_df["log2FoldChange"],
            "SE": subset_df["lfcSE"],
            "lower_ci": subset_df["log2FoldChange"] - ci_multiplier * subset_df["lfcSE"],
            "upper_ci": subset_df["log2FoldChange"] + ci_multiplier * subset_df["lfcSE"],
            "pct_contribution": contribution,
        })

        forest_df = forest_df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["L2FC", "SE", "pct_contribution"]
        )

        if cutoff is not None:
            keep_genes = (
                forest_df
                .sort_values("pct_contribution", ascending=False)
                .assign(cum_contribution=lambda df: df["pct_contribution"].cumsum())
                .loc[lambda df: df["cum_contribution"].le(cutoff), "gene"]
            )

            if len(keep_genes) == 0:
                keep_genes = (
                    forest_df
                    .sort_values("pct_contribution", ascending=False)
                    .head(1)["gene"]
                )

            forest_df = forest_df.loc[forest_df["gene"].isin(keep_genes)]

        forest_df = forest_df.sort_values("pct_contribution", ascending=True)

        n_genes = len(forest_df)

        if figsize is None:
            figsize = (11, max(5, 0.32 * n_genes + 2.5))

        fig, ax = plt.subplots(
            nrows=2,
            ncols=2,
            gridspec_kw={
                "width_ratios": [2.4, 1],
                "height_ratios": [max(4, n_genes * 0.3), 1],
                "hspace": 0.2,
                "wspace": 0.05,
            },
            figsize=figsize,
            sharex="col",
        )

        forest_ax = ax[0, 0]
        contrib_ax = ax[0, 1]
        meta_ax = ax[1, 0]
        empty_ax = ax[1, 1]

        y = np.arange(n_genes)

        colors = np.where(forest_df["L2FC"].ge(0), "firebrick", "steelblue")

        forest_ax.errorbar(
            x=forest_df["L2FC"],
            y=y,
            xerr=ci_multiplier * forest_df["SE"],
            fmt="none",
            ecolor="0.35",
            elinewidth=1.2,
            capsize=3,
            zorder=1,
        )

        forest_ax.scatter(
            x=forest_df["L2FC"],
            y=y,
            s=point_size,
            c=colors,
            edgecolor="black",
            linewidth=0.4,
            zorder=2,
        )

        forest_ax.axvline(0, color="black", linestyle=":", linewidth=1.2)
        forest_ax.set_yticks(y)
        forest_ax.set_yticklabels(forest_df["gene"])
        forest_ax.set_ylabel("")
        forest_ax.set_xlabel("log2 fold change")
        forest_ax.grid(axis="x", alpha=0.25)
        forest_ax.spines["top"].set_visible(False)
        forest_ax.spines["right"].set_visible(False)

        contrib_ax.barh(
            y=y,
            width=forest_df["pct_contribution"],
            color="0.45",
            edgecolor="black",
            linewidth=0.3,
        )

        contrib_ax.set_yticks(y)
        contrib_ax.set_yticklabels([])
        contrib_ax.set_xlabel("Contribution")
        contrib_ax.grid(axis="x", alpha=0.25)
        contrib_ax.spines["top"].set_visible(False)
        contrib_ax.spines["right"].set_visible(False)
        contrib_ax.spines["left"].set_visible(False)
        contrib_ax.tick_params(axis="y", length=0)

        if contribution_label:
            xmax = forest_df["pct_contribution"].max()
            for yi, value in zip(y, forest_df["pct_contribution"]):
                contrib_ax.text(
                    value + xmax * 0.02,
                    yi,
                    f"{value:.1%}",
                    va="center",
                    fontsize=8,
                )
            contrib_ax.set_xlim(0, xmax * 1.25)

        meta_lfc = meta_estimation["metaLFC"]
        meta_se = meta_estimation["metaSE"]

        meta_ax.errorbar(
            x=meta_lfc,
            y=[0],
            xerr=ci_multiplier * meta_se,
            fmt="none",
            ecolor="0.35",
            elinewidth=1.4,
            capsize=4,
            zorder=1,
        )

        meta_color = "firebrick" if meta_lfc >= 0 else "steelblue"

        meta_ax.scatter(
            x=[meta_lfc],
            y=[0],
            s=70,
            c=meta_color,
            edgecolor="black",
            linewidth=0.5,
            zorder=2,
        )

        meta_ax.axvline(0, color="black", linestyle=":", linewidth=1.2)
        meta_ax.set_yticks([0])
        meta_ax.set_yticklabels(["Meta-estimate"])
        meta_ax.set_xlabel("meta log2 fold change")
        meta_ax.grid(axis="x", alpha=0.25)
        meta_ax.spines["top"].set_visible(False)
        meta_ax.spines["right"].set_visible(False)

        empty_ax.axis("off")

        if title is None:
            title = term

        fig.suptitle(
            title,
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        displayed_contribution = forest_df["pct_contribution"].sum()

        fig.text(
            0.01,
            0.01,
            f"Displayed genes: {n_genes} | Displayed contribution: {displayed_contribution:.1%}",
            fontsize=9,
            color="0.35",
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        return forest_df, fig, ax