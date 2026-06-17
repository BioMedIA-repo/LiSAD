import os
import numpy as np
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist


def sample_few_shot_data(adata, k_shot, folder, seed=2024, n_neighbors_check=15):
    """
    Select few-shot training spots for each class.

    The selection strategy first keeps spots with high local label consistency,
    and then applies farthest-point sampling to improve spatial diversity.
    """
    raw_labels = adata.obs["ground_truth"].astype(str).values

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(raw_labels)
    labels_tensor = torch.tensor(labels_encoded, dtype=torch.long)

    coords = adata.obsm["spatial"]
    num_nodes = len(labels_encoded)
    num_classes = len(label_encoder.classes_)

    n_neighbors_check = min(n_neighbors_check, num_nodes - 1)

    print(
        f"\n>>> Selecting {k_shot} high-purity and spatially diverse spots per class..."
    )

    nbrs = NearestNeighbors(n_neighbors=n_neighbors_check + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    neighbor_indices = indices[:, 1:]

    purity_scores = np.zeros(num_nodes)

    for i in range(num_nodes):
        self_label = labels_encoded[i]
        neighbor_labels = labels_encoded[neighbor_indices[i]]
        purity_scores[i] = np.sum(neighbor_labels == self_label) / n_neighbors_check

    train_indices_list = []

    for class_id in range(num_classes):
        class_indices = np.where(labels_encoded == class_id)[0]

        if len(class_indices) == 0:
            continue

        class_purity = purity_scores[class_indices]
        purity_threshold = np.percentile(class_purity, 50)

        candidate_mask = class_purity >= purity_threshold
        candidate_indices = class_indices[candidate_mask]

        if len(candidate_indices) <= k_shot:
            selected = candidate_indices
        else:
            candidate_coords = coords[candidate_indices]
            selected_sub_idx = [0]

            for _ in range(k_shot - 1):
                dist_to_selected = cdist(
                    candidate_coords,
                    candidate_coords[selected_sub_idx],
                    metric="euclidean",
                )
                min_dist_to_selected = np.min(dist_to_selected, axis=1)
                next_idx = np.argmax(min_dist_to_selected)
                selected_sub_idx.append(next_idx)

            selected = candidate_indices[selected_sub_idx]

        train_indices_list.extend(selected)

        print(
            f" - Class {label_encoder.classes_[class_id]}: "
            f"Selected {len(selected)} spots."
        )

    train_idx = torch.tensor(train_indices_list, dtype=torch.long)

    rng = np.random.default_rng(seed)
    train_idx = train_idx[rng.permutation(len(train_idx))]

    train_labels = labels_tensor[train_idx]

    all_indices = np.arange(num_nodes)
    remaining_indices = np.setdiff1d(all_indices, np.array(train_indices_list))

    test_idx = torch.tensor(remaining_indices, dtype=torch.long)
    test_labels = labels_tensor[test_idx]

    os.makedirs(folder, exist_ok=True)

    torch.save(train_idx, os.path.join(folder, "train_idx.pt"))
    torch.save(train_labels, os.path.join(folder, "train_labels.pt"))
    torch.save(test_idx, os.path.join(folder, "test_idx.pt"))
    torch.save(test_labels, os.path.join(folder, "test_labels.pt"))

    np.save(
        os.path.join(folder, "label_encoder_classes.npy"),
        label_encoder.classes_,
    )

    return train_idx, train_labels, test_idx, test_labels, num_classes