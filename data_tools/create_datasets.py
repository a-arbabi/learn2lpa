import data_tools.data_utils as data_utils
import pickle
from pathlib import Path
import numpy as np
import argparse

def get_train_valid_test_split_ids(n_nodes, train_frac, valid_frac, test_frac):
    np.random.seed(1234)
    node_ids = np.random.permutation(n_nodes)
    frac_sum = train_frac + valid_frac + test_frac
    if frac_sum != 1.0:
        raise ValueError(
            "The fold fractions do not sum to 1.0 (sum=={}).".format(frac_sum)
        )
    split_point_1 = int(n_nodes * train_frac)
    split_point_2 = int(n_nodes * (train_frac + valid_frac))
    fold_ids = {
        "train": node_ids[0:split_point_1],
        "valid": node_ids[split_point_1:split_point_2],
        "test": node_ids[split_point_2:],
    }
    return fold_ids


def prepare_and_save_ppi_hpo_dataset(raw_data_dir, target_data_dir, network_name, validation_type):
    if validation_type != "temporal" and validation_type != "cv":
        raise ValueError(f'Unkown validation_type: {validation_type}')
    if network_name != "string" and network_name != "genemania":
        raise ValueError(f'Unkown network_name: {network_name}')

    print(f"validation_type: {validation_type}")
    is_temporal = (validation_type == "temporal")
    (
        prot2hp_matrix,
        hp2ancestors_matrix,
        graph,
        protids_by_fold,
        id2prot,
    ) = data_utils.get_data_matrices(
        raw_data_dir,
        is_temporal=is_temporal,
        validate_prots_with_hpolabeler=False,
        filter_genes_with_multi_prots=True,
        filter_genes_without_valid_annotation=True,
        use_full_ppi=False,
        network_name=network_name
    )
    del hp2ancestors_matrix
    del id2prot

    labels = prot2hp_matrix

    if not is_temporal:
        n_nodes = labels.shape[0]
        fold_ids = get_train_valid_test_split_ids(n_nodes, 0.8, 0.1, 0.1)
    else:
        fold_ids = protids_by_fold

    target_data_path = Path(target_data_dir)
    target_data_path.mkdir(parents=True, exist_ok=True)

    with open(target_data_path / "graph.p", "wb") as f:
        pickle.dump(graph, file=f)
    with open(target_data_path / "labels.p", "wb") as f:
        pickle.dump(labels, file=f)
    with open(target_data_path / "fold_ids.p", "wb") as f:
        pickle.dump(fold_ids, file=f)

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess datasets and save to file."
    )
    parser.add_argument("input_data_dir", help="Path to the raw data directory.")
    parser.add_argument("output_data_dir", help="Path to the output directory for saving preprocessed data.")
    args = parser.parse_args()
    #for network_name in ["string", "genemania"]:
    for network_name in ["string"]:
        for validation_type in ["cv", "temporal"]:
            output_dir_path = Path(args.output_data_dir) / f"ppi_hpo_{network_name}_{validation_type}/"
            prepare_and_save_ppi_hpo_dataset(args.input_data_dir, output_dir_path, network_name, validation_type)


if __name__ == "__main__":
    main()