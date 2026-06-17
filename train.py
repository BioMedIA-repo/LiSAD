import os
import random
import datetime

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import MinMaxScaler

from model import HGNNClassifier
from sample_few_shot_data import sample_few_shot_data
from adata_processing import LoadSingle10xAdata, LoadSingleAdata, compute_agf_metadata


CONFIG = {
    "gpu_id": 0,
    "seed": 2024,

    "data_path": "./data/DLPFC/151507",
    "dataset_type": "10x",
    "save_dir": "./results/151507",
    "log_file": "./results/151507/train_log.txt",

    "n_top_genes": 3000,
    "n_neighbors": 10,
    "use_pca": True,
    "n_pca_components": 128,
    "is_probH": True,
    "m_prob": 1.0,

    "k_shot": 5,
    "epochs": 500,
    "lr": 0.002,
    "weight_decay": 1e-3,

    "n_hidden": 128,
    "n_embed": 64,
    "dropout": 0.5,

    "alpha_rec": 1.0,
    "alpha_tv": 0.01,
    "omega_0": 1.0,
}


class SpatialTVLoss(nn.Module):
    def __init__(self, indices, w_cos, w_sin):
        super().__init__()
        self.indices = indices
        self.w_cos = w_cos
        self.w_sin = w_sin

    def forward(self, x_rec):
        neighbor_x = x_rec[self.indices]
        diff = neighbor_x - x_rec.unsqueeze(1)

        grad_x = torch.einsum("nk,nkd->nd", self.w_cos, diff)
        grad_y = torch.einsum("nk,nkd->nd", self.w_sin, diff)

        std_x = torch.std(grad_x, dim=0, keepdim=True) + 1e-6
        std_y = torch.std(grad_y, dim=0, keepdim=True) + 1e-6

        loss_tv = torch.mean(torch.abs(grad_x / std_x) + torch.abs(grad_y / std_y))
        return loss_tv


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(gpu_id):
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double().sum()
    return correct / len(labels)


def save_result_to_txt(
    config,
    final_acc,
    final_ari,
    final_loss,
    ari_raw,
    ari_refined,
    nmi_raw,
    nmi_refined,
):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_content = []
    log_content.append(f"[{timestamp}] Training Record")
    log_content.append("-" * 40)

    for key, value in config.items():
        log_content.append(f"{key}: {value}")

    log_content.append("-" * 40)
    log_content.append("FINAL RESULT")
    log_content.append(f"Final Test Accuracy: {final_acc:.4f}")
    log_content.append(f"Final ARI         : {final_ari:.4f}")
    log_content.append(f"Raw ARI           : {ari_raw:.4f}")
    log_content.append(f"Refined ARI       : {ari_refined:.4f}")
    log_content.append(f"Raw NMI           : {nmi_raw:.4f}")
    log_content.append(f"Refined NMI       : {nmi_refined:.4f}")
    log_content.append(f"Final Train Loss  : {final_loss:.4f}")
    log_content.append("=" * 60 + "\n")

    log_dir = os.path.dirname(config["log_file"])
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    with open(config["log_file"], "a", encoding="utf-8") as f:
        f.write("\n".join(log_content))

    print(f"\n[Info] Result saved to {config['log_file']}")


def refine_label(adata, preds, n_neighbors=15):
    import scipy.stats as stats

    print(f">>> Running spatial refinement with k={n_neighbors}...")

    tmp_adata = adata.copy()
    sc.pp.neighbors(tmp_adata, n_neighbors=n_neighbors, use_rep="spatial")
    connectivities = tmp_adata.obsp["connectivities"]

    refined_preds = []

    for i in range(len(preds)):
        neighbors = connectivities[i, :].nonzero()[1]
        neighbors = np.append(neighbors, i)

        neighbor_preds = preds[neighbors]
        mode_res = stats.mode(neighbor_preds, keepdims=True)
        refined_preds.append(mode_res.mode[0])

    return np.array(refined_preds)


def load_dataset(config):
    if config["dataset_type"] == "10x":
        loader = LoadSingle10xAdata(
            path=config["data_path"],
            n_top_genes=config["n_top_genes"],
            n_neighbors=config["n_neighbors"],
            seed=config["seed"],
            is_probH=config["is_probH"],
            m_prob=config["m_prob"],
            n_pca_components=config["n_pca_components"],
            use_pca=config["use_pca"],
        )
    elif config["dataset_type"] == "h5ad":
        loader = LoadSingleAdata(
            path=config["data_path"],
            n_neighbors=config["n_neighbors"],
            seed=config["seed"],
            is_probH=config["is_probH"],
            m_prob=config["m_prob"],
            n_pca_components=config["n_pca_components"],
            use_pca=config["use_pca"],
            n_top_genes=config["n_top_genes"],
            label=True,
        )
    else:
        raise ValueError("dataset_type must be either '10x' or 'h5ad'.")

    return loader.run()


def main():
    setup_seed(CONFIG["seed"])
    device = get_device(CONFIG["gpu_id"])

    print(f"Running on: {device}")
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    print(f">>> Loading data from {CONFIG['data_path']}...")
    adata = load_dataset(CONFIG)

    print(">>> Preparing spatial TV metadata...")
    adata = compute_agf_metadata(adata, k_geom=2 * CONFIG["n_neighbors"])

    agf_indices = torch.LongTensor(adata.obsm["agf_indices"]).to(device)
    agf_w_cos = torch.FloatTensor(adata.obsm["agf_w_cos"]).to(device)
    agf_w_sin = torch.FloatTensor(adata.obsm["agf_w_sin"]).to(device)

    gene_target_np = adata.obsm["feat"]
    gene_target = torch.FloatTensor(gene_target_np).to(device)

    spatial_raw = adata.obsm["spatial"]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    spatial_norm = scaler.fit_transform(spatial_raw)
    coords = torch.FloatTensor(spatial_norm).to(device)

    G_np = adata.obsm["hypergraph_laplacian"]
    G = torch.FloatTensor(np.array(G_np)).to(device)

    print(f"Coords shape: {coords.shape}")
    print(f"Gene target shape: {gene_target.shape}")
    print(f"Hypergraph matrix shape: {G.shape}")

    split_folder = os.path.join(CONFIG["save_dir"], "splits")

    print(f">>> Sampling few-shot data with k={CONFIG['k_shot']}...")
    train_idx, train_labels, test_idx, test_labels, num_classes = sample_few_shot_data(
        adata=adata,
        k_shot=CONFIG["k_shot"],
        folder=split_folder,
        seed=CONFIG["seed"],
    )

    train_idx = train_idx.to(device)
    train_labels = train_labels.to(device)
    test_idx = test_idx.to(device)
    test_labels = test_labels.to(device)

    model = HGNNClassifier(
        spatial_dim=coords.shape[1],
        n_genes=gene_target.shape[1],
        n_hid=CONFIG["n_hidden"],
        n_embed=CONFIG["n_embed"],
        n_class=num_classes,
        omega_0=CONFIG["omega_0"],
        dropout=CONFIG["dropout"],
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["epochs"],
        eta_min=1e-5,
    )

    criterion_cls = nn.CrossEntropyLoss()
    criterion_rec = nn.MSELoss()
    criterion_tv = SpatialTVLoss(agf_indices, agf_w_cos, agf_w_sin)

    print(">>> Starting training...")

    final_loss = 0.0
    final_acc = 0.0
    final_ari = 0.0

    final_model_path = os.path.join(CONFIG["save_dir"], "final_model.pth")

    for epoch in range(CONFIG["epochs"]):
        model.train()
        optimizer.zero_grad()

        logits, x_rec, _ = model(coords, G)

        loss_cls = criterion_cls(logits[train_idx], train_labels)
        loss_rec = criterion_rec(x_rec, gene_target)
        loss_tv = criterion_tv(x_rec)

        loss = loss_cls + CONFIG["alpha_rec"] * loss_rec + CONFIG["alpha_tv"] * loss_tv

        loss.backward()
        optimizer.step()
        scheduler.step()

        final_loss = loss.item()

        model.eval()
        with torch.no_grad():
            logits, _, _ = model(coords, G)
            preds = logits.max(1)[1]

            acc_train = accuracy(logits[train_idx], train_labels)
            acc_test = accuracy(logits[test_idx], test_labels)

            current_preds_np = preds.cpu().numpy()
            current_ari = adjusted_rand_score(
                adata.obs["ground_truth"],
                current_preds_np,
            )
            current_nmi = normalized_mutual_info_score(
                adata.obs["ground_truth"],
                current_preds_np,
            )

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:03d} | "
                f"Loss: {loss.item():.4f} | "
                f"Train Acc: {acc_train.item():.4f} | "
                f"Test Acc: {acc_test.item():.4f} | "
                f"ARI: {current_ari:.4f} | "
                f"NMI: {current_nmi:.4f}"
            )

    torch.save(model.state_dict(), final_model_path)
    print("-" * 50)
    print(f"Training finished after {CONFIG['epochs']} epochs.")
    print(f"Final model saved to: {final_model_path}")

    model.eval()
    with torch.no_grad():
        final_logits, _, _ = model(coords, G)

    preds_raw = final_logits.detach().cpu().max(1)[1].numpy()

    final_acc = accuracy(final_logits[test_idx], test_labels).item()
    ari_raw = adjusted_rand_score(adata.obs["ground_truth"], preds_raw)
    nmi_raw = normalized_mutual_info_score(adata.obs["ground_truth"], preds_raw)
    final_ari = ari_raw

    preds_refined = refine_label(adata, preds_raw, n_neighbors=20)

    ari_refined = adjusted_rand_score(adata.obs["ground_truth"], preds_refined)
    nmi_refined = normalized_mutual_info_score(adata.obs["ground_truth"], preds_refined)

    print("\n[Final Result Analysis]")
    print(f"Raw ARI     : {ari_raw:.4f}")
    print(f"Refined ARI : {ari_refined:.4f} ({ari_refined - ari_raw:+.4f})")
    print(f"Raw NMI     : {nmi_raw:.4f}")
    print(f"Refined NMI : {nmi_refined:.4f} ({nmi_refined - nmi_raw:+.4f})")

    adata.obs["HGNN_Raw"] = preds_raw.astype(str)
    adata.obs["HGNN_Refined"] = preds_refined.astype(str)

    adata.obs["split"] = "test"
    adata.obs.iloc[
        train_idx.cpu().numpy(),
        adata.obs.columns.get_loc("split"),
    ] = "train"

    result_path = os.path.join(CONFIG["save_dir"], "result_adata.h5ad")
    adata.write(result_path)

    print(f"Results saved to: {result_path}")

    save_result_to_txt(
        CONFIG,
        final_acc,
        final_ari,
        final_loss,
        ari_raw,
        ari_refined,
        nmi_raw,
        nmi_refined,
    )


if __name__ == "__main__":
    main()