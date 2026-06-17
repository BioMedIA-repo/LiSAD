import os

import numpy as np
import pandas as pd
import scanpy as sc
import ot

from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def generate_G_from_H(H, variable_weight=False):
    H = np.asarray(H)
    n_edge = H.shape[1]

    W = np.ones(n_edge)
    DV = np.sum(H * W, axis=1)
    DE = np.sum(H, axis=0)

    eps = 1e-8
    invDE = np.diag(np.power(DE.astype(float) + eps, -1))
    invDV = np.diag(np.power(DV.astype(float) + eps, -1))
    W = np.diag(W)

    if variable_weight:
        invDV_H = invDV @ H
        invDE_HT = invDE @ H.T
        return invDV_H, W, invDE_HT

    G = invDV @ H @ W @ invDE @ H.T
    return G


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    n_obj = dis_mat.shape[0]
    k_neig = min(k_neig, n_obj)

    H = np.zeros((n_obj, n_obj))

    for center_idx in range(n_obj):
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()

        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                avg_dis = np.average(dis_vec)
                if avg_dis == 0:
                    avg_dis = 1.0
                H[node_idx, center_idx] = np.exp(
                    -dis_vec[node_idx] ** 2 / (m_prob * avg_dis) ** 2
                )
            else:
                H[node_idx, center_idx] = 1.0

    return H


def construct_H_refined(
    spatial_coords,
    gene_features,
    final_hyperedge_k=4,
    is_probH=True,
    m_prob=1,
):
    print(f"Building spatial hypergraph with k={final_hyperedge_k}...")
    spatial_dist_mat = ot.dist(spatial_coords, spatial_coords, metric="euclidean")

    print(f"Building feature hypergraph with k={final_hyperedge_k}...")
    gene_dist_mat = ot.dist(gene_features, gene_features, metric="cosine")

    H_spatial = construct_H_with_KNN_from_distance(
        spatial_dist_mat,
        k_neig=final_hyperedge_k,
        is_probH=is_probH,
        m_prob=m_prob,
    )

    H_feature = construct_H_with_KNN_from_distance(
        gene_dist_mat,
        k_neig=final_hyperedge_k,
        is_probH=is_probH,
        m_prob=m_prob,
    )

    print("Concatenating spatial and feature hypergraphs...")
    H_combined = np.concatenate((H_spatial, H_feature), axis=1)

    return H_combined


def construct_H_spatial(spatial_coords, k_neig=10, is_probH=True, m_prob=1):
    print(f"Building spatial-only hypergraph with k={k_neig}...")
    spatial_dist_mat = ot.dist(spatial_coords, spatial_coords, metric="euclidean")

    H = construct_H_with_KNN_from_distance(
        spatial_dist_mat,
        k_neig=k_neig,
        is_probH=is_probH,
        m_prob=m_prob,
    )

    return H


def compute_agf_metadata(adata, k_geom=15):
    coords = adata.obsm["spatial"]

    nbrs = NearestNeighbors(n_neighbors=k_geom + 1).fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    indices = indices[:, 1:]
    distances = distances[:, 1:]

    adj_coords = coords[indices]
    rel_coords = adj_coords - coords[:, np.newaxis, :]

    dx = rel_coords[:, :, 0]
    dy = rel_coords[:, :, 1]
    phi = np.arctan2(dy, dx)

    sigma = distances[:, k_geom // 2]
    sigma_sq = (sigma ** 2).reshape(-1, 1) + 1e-8

    gamma = np.exp(-(distances ** 2) / sigma_sq)
    gamma = gamma / (gamma.sum(axis=1, keepdims=True) + 1e-8)

    adata.obsm["agf_indices"] = indices
    adata.obsm["agf_w_cos"] = (gamma * np.cos(phi)).astype(np.float32)
    adata.obsm["agf_w_sin"] = (gamma * np.sin(phi)).astype(np.float32)

    return adata


class LoadSingle10xAdata:
    def __init__(
        self,
        path: str,
        n_top_genes: int = 3000,
        n_neighbors: int = 4,
        seed: int = 100,
        image_emb: bool = True,
        image_pca: int = 64,
        is_probH: bool = False,
        m_prob: float = 1.0,
        n_pca_components: int = 50,
        label: bool = True,
        filter_na: bool = True,
        use_pca: bool = False,
    ):
        self.path = path
        self.n_top_genes = n_top_genes
        self.n_neighbors = n_neighbors
        self.adata = None
        self.label = label
        self.filter_na = filter_na
        self.is_probH = is_probH
        self.m_prob = m_prob
        self.n_pca_components = n_pca_components
        self.use_pca = use_pca
        self.seed = seed

    def load_data(self):
        self.adata = sc.read_visium(
            self.path,
            count_file="filtered_feature_bc_matrix.h5",
            load_images=True,
        )
        self.adata.var_names_make_unique()

    def preprocess(self):
        sc.pp.highly_variable_genes(
            self.adata,
            flavor="seurat_v3",
            n_top_genes=self.n_top_genes,
        )
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        self.adata = self.adata[:, self.adata.var["highly_variable"]].copy()

    def generate_gene_expr(self):
        sc.pp.scale(self.adata, zero_center=True, max_value=10)

        if isinstance(self.adata.X, (csc_matrix, csr_matrix)):
            feat = self.adata.X.toarray()
        else:
            feat = self.adata.X

        self.adata.obsm["feat"] = feat

        if self.use_pca:
            print(f"Computing PCA features with n_components={self.n_pca_components}...")
            pca = PCA(n_components=self.n_pca_components, random_state=self.seed)
            feat_pca = pca.fit_transform(feat)
            self.adata.obsm["feat_pca"] = feat_pca
        else:
            self.adata.obsm["feat_pca"] = feat

    def construct_hypergraph(self):
        spatial_coords = self.adata.obsm["spatial"]

        H = construct_H_spatial(
            spatial_coords=spatial_coords,
            k_neig=self.n_neighbors,
            is_probH=self.is_probH,
            m_prob=self.m_prob,
        )

        print("Generating hypergraph propagation matrix...")
        G = generate_G_from_H(H)

        self.adata.obsm["hypergraph_incidence"] = H
        self.adata.obsm["hypergraph_laplacian"] = G

        print("Hypergraph construction completed.")

    def load_label(self):
        label_path = os.path.join(self.path, "truth.txt")
        df_meta = pd.read_csv(label_path, sep="\t", header=None)
        self.adata.obs["ground_truth"] = df_meta[1].values

        if self.filter_na:
            self.adata = self.adata[
                ~pd.isnull(self.adata.obs["ground_truth"])
            ].copy()

    def run(self):
        self.load_data()

        if self.label:
            self.load_label()

        self.preprocess()
        self.generate_gene_expr()
        self.construct_hypergraph()

        print("AnnData preprocessing completed.")

        return self.adata


class LoadSingleAdata:
    def __init__(
        self,
        path: str,
        n_neighbors: int = 3,
        is_probH: bool = False,
        m_prob: float = 1.0,
        label: bool = False,
        n_pca_components: int = 50,
        image_emb: bool = True,
        image_pca: int = 64,
        filter_na: bool = True,
        n_top_genes: int = 161,
        use_pca: bool = False,
        seed: int = 100,
    ):
        self.path = path
        self.n_neighbors = n_neighbors
        self.adata = None
        self.label = label
        self.filter_na = filter_na
        self.n_top_genes = n_top_genes
        self.is_probH = is_probH
        self.m_prob = m_prob
        self.n_pca_components = n_pca_components
        self.use_pca = use_pca
        self.image_emb = image_emb
        self.image_pca = image_pca
        self.seed = seed

    def load_data(self):
        self.adata = sc.read_h5ad(self.path)
        self.adata.var_names_make_unique()

    def preprocess(self):
        sc.pp.highly_variable_genes(
            self.adata,
            flavor="seurat_v3",
            n_top_genes=self.n_top_genes,
        )
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        self.adata = self.adata[:, self.adata.var["highly_variable"]].copy()

    def generate_gene_expr(self):
        sc.pp.scale(self.adata, zero_center=True, max_value=10)

        if isinstance(self.adata.X, (csc_matrix, csr_matrix)):
            feat = self.adata.X.toarray()
        else:
            feat = self.adata.X

        self.adata.obsm["feat"] = feat

        if self.use_pca:
            print(f"Computing PCA features with n_components={self.n_pca_components}...")
            pca = PCA(n_components=self.n_pca_components, random_state=self.seed)
            feat_pca = pca.fit_transform(feat)
            self.adata.obsm["feat_pca"] = feat_pca
        else:
            self.adata.obsm["feat_pca"] = feat

    def construct_hypergraph(self):
        spatial_coords = self.adata.obsm["spatial"]

        H = construct_H_spatial(
            spatial_coords=spatial_coords,
            k_neig=self.n_neighbors,
            is_probH=self.is_probH,
            m_prob=self.m_prob,
        )

        print("Generating hypergraph propagation matrix...")
        G = generate_G_from_H(H)

        self.adata.obsm["hypergraph_incidence"] = H
        self.adata.obsm["hypergraph_laplacian"] = G

        print("Hypergraph construction completed.")

    def load_label(self):
        if "ground_truth" not in self.adata.obs:
            raise ValueError("'ground_truth' column was not found in adata.obs.")

        if self.filter_na:
            self.adata = self.adata[
                ~pd.isnull(self.adata.obs["ground_truth"])
            ].copy()

    def run(self):
        self.load_data()

        if self.label:
            self.load_label()

        self.preprocess()
        self.generate_gene_expr()
        self.construct_hypergraph()

        print("AnnData preprocessing completed.")

        return self.adata