import os

import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib import rcParams


H5AD_PATH = "./results/151507/result_adata.h5ad"
SAVE_PLOT_PATH = "./results/151507/comparison_result.png"

PLOT_KEY = "HGNN_Raw"
SPOT_SIZE = 100
FIG_SIZE = (8, 6)

PLOT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7",
    "#dbdb8d", "#9edae5", "#393b79", "#637939", "#8c6d31", "#843c39",
]

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"] + rcParams["font.serif"]


def visualize_results(
    h5ad_path=H5AD_PATH,
    save_plot_path=SAVE_PLOT_PATH,
    key=PLOT_KEY,
):
    if not os.path.exists(h5ad_path):
        print(f"Error: result file not found: {h5ad_path}")
        print("Please run train.py first.")
        return

    print(f">>> Loading result from {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)

    if key not in adata.obs:
        print(f"Error: {key} was not found in adata.obs.")
        return

    print(">>> Preparing plot...")

    adata.obs[key] = adata.obs[key].astype("category")

    n_cats = len(adata.obs[key].cat.categories)
    new_cats = [str(i + 1) for i in range(n_cats)]
    adata.obs[key] = adata.obs[key].cat.rename_categories(new_cats)

    print(f"Categories were renamed to: {new_cats}")

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    sc.pl.spatial(
        adata,
        color=key,
        ax=ax,
        img_key=None,
        spot_size=SPOT_SIZE,
        palette=PLOT_COLORS,
        alpha=1.0,
        legend_loc=None,
        show=False,
        title="",
        frameon=False,
    )

    plt.axis("off")

    save_dir = os.path.dirname(save_plot_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_plot_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f">>> Visualization saved to: {save_plot_path}")


if __name__ == "__main__":
    visualize_results()