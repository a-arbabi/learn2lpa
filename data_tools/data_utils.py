import numpy as np
from pathlib import Path
from pronto import Ontology
import random
import tensorflow as tf
import pandas as pd
import pickle


def file2dict(
    path,
    key_cols,
    val_cols,
    pass_header=True,
    separator="\t",
):
    if pass_header:
        skiprows = 1
    else:
        skiprows = 0
    df = pd.read_csv(path, sep=separator, header=None, skiprows=skiprows)

    if isinstance(key_cols, int):
        key_cols_names = df.columns[key_cols]
    else:
        key_cols_names = [df.columns[idx] for idx in key_cols]
    if isinstance(val_cols, int):
        val_cols_names = df.columns[val_cols]
    else:
        val_cols_names = [df.columns[idx] for idx in val_cols]

    data_dict = (
        df.groupby(key_cols_names)
        .apply(lambda grp: tuple(grp[val_cols_names].values.tolist()))
        .to_dict()
    )
    return data_dict


def get_temporal_prot2hps(
    data_path, gene2prot, valid_hps, filter_genes_without_valid_annotation
):
    temporal_files = {
        "train": data_path
        / "HPOLabeler_data/annotation/ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype_2017_02_24.txt",
        "valid": data_path
        / "HPOLabeler_data/annotation/ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype_2018_03_09.txt",
        "test": data_path
        / "HPOLabeler_data/annotation/ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype_2018_12_21.txt",
    }
    temporal_gene2hps = {
        snap: file2dict(temporal_files[snap], 0, 3) for snap in temporal_files
    }
    prot2hps = {}
    prots_by_fold = {fold: set() for fold in temporal_gene2hps}
    for fold in ["train", "valid", "test"]:
        for gene, hp_list in temporal_gene2hps[fold].items():
            hp_list = {hp for hp in hp_list if hp in valid_hps}
            if filter_genes_without_valid_annotation and len(hp_list) == 0:
                continue
            if gene not in gene2prot:
                continue
            for prot in gene2prot[gene]:
                if prot in prot2hps:  # prot has been seen in a previous fold
                    continue
                prot2hps[prot] = hp_list
                prots_by_fold[fold].add(prot)
    return prot2hps, prots_by_fold


def get_cv_prot2hps(
    data_path, gene2prot, valid_hps, filter_genes_without_valid_annotation
):
    gene2hps = file2dict(
        data_path
        / "HPOLabeler_data/annotation/ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype_2018_07_27.txt",
        0,
        3,
    )
    prot2hps = {}
    for gene, hp_list in gene2hps.items():
        hp_list = {hp for hp in hp_list if hp in valid_hps}
        if filter_genes_without_valid_annotation and len(hp_list) == 0:
            continue
        if gene not in gene2prot:
            continue
        for prot in gene2prot[gene]:
            prot2hps[prot] = hp_list
    prots_list = list(prot2hps.keys())
    random.seed(123)
    random.shuffle(prots_list)
    prots_by_fold = {"cv_prots": prots_list}
    return prot2hps, prots_by_fold


def get_data_dicts(
    data_path,
    is_temporal,
    validate_prots_with_hpolabeler=True,
    filter_genes_with_multi_prots=True,
    filter_genes_without_valid_annotation=True,
    network_name="string",
):
    data_path = Path(data_path)
    valid_prots_files = [
        path for path in (data_path / "HPOLabeler_data/feature/temporal/").iterdir()
    ]
    gene2prot_file = data_path / "gene2prot.txt"
    string2prot_file = data_path / "string2prot.txt"
    ensembl2prot_file = data_path / "ensembl2prot.txt"
    if is_temporal:
        hpo_file = data_path / "hp_2017_02_24.obo"
    else:
        hpo_file = data_path / "hp_2018_07_25.obo"
    ppi_string_file = data_path / "9606.protein.links.v10.5.txt"
    ppi_genemania_file = data_path / "COMBINED.DEFAULT_NETWORKS.BP_COMBINING.txt"

    # Read files
    gene2prot = file2dict(gene2prot_file, 0, 2)
    cl = Ontology(hpo_file)
    if network_name=="string":
        ppi_string = file2dict(
            ppi_string_file,
            0,
            (1, 2),
            separator=" ",
        )
        string2prot = file2dict(string2prot_file, 0, 1)
    elif network_name=="genemania":
        ppi_genemania = file2dict(
            ppi_genemania_file,
            0,
            (1, 2),
            separator="\t",
        )
        ensembl2prot = file2dict(ensembl2prot_file, 0, 1)
    else:
        raise ValueError(f"Unkown network name: {network_name}")
    valid_prots = set()
    if validate_prots_with_hpolabeler:
        for valid_prots_file in valid_prots_files:
            with open(valid_prots_file) as f:
                valid_prots.update({line.split()[0] for line in f})

    # Filter prots
    if validate_prots_with_hpolabeler:
        gene2prot = {
            gene: {prot for prot in gene2prot[gene] if prot in valid_prots}
            for gene in gene2prot
        }
    if filter_genes_with_multi_prots:
        gene2prot = {
            gene: prots for gene, prots in gene2prot.items() if len(prots) == 1
        }

    # Create prot2hps dictionary and separate prots by train/valid/test folds
    valid_hps = {x.id for x in cl["HP:0000118"].subclasses()}
    if is_temporal:
        prot2hps, prots_by_fold = get_temporal_prot2hps(
            data_path, gene2prot, valid_hps, filter_genes_without_valid_annotation
        )
    else:
        prot2hps, prots_by_fold = get_cv_prot2hps(
            data_path, gene2prot, valid_hps, filter_genes_without_valid_annotation
        )

    # Create ppi network
    ppi_network = {}
    if network_name=="string":
        for string, string_links in ppi_string.items():
            if string in string2prot and len(string2prot[string]) == 1:
                prot = next(iter(string2prot[string]))
                ppi_network[prot] = [
                    (next(iter(string2prot[string_neighbour])), int(score) / 1000)
                    for string_neighbour, score in string_links
                    if string_neighbour in string2prot
                    and len(string2prot[string_neighbour]) == 1
                ]
    elif network_name=="genemania":
        for ensembl, ensembl_links in ppi_genemania.items():
            if ensembl in ensembl2prot and len(ensembl2prot[ensembl]) == 1:
                prot = next(iter(ensembl2prot[ensembl]))
                ppi_network[prot] = [
                    (next(iter(ensembl2prot[ensembl_neighbour])), float(score))
                    for ensembl_neighbour, score in ensembl_links
                    if ensembl_neighbour in ensembl2prot
                    and len(ensembl2prot[ensembl_neighbour]) == 1
                ]
    # Create hpo ancestary network
    direct_hps = {hp for hps in prot2hps.values() for hp in hps}
    relevant_hps = {
        anc_hp.id
        for direct_hp in direct_hps
        for anc_hp in cl[direct_hp].superclasses()
        if anc_hp.id in valid_hps
    }
    hp2ancestors = {
        hp: {anc_hp.id for anc_hp in cl[hp].superclasses() if anc_hp.id in valid_hps}
        for hp in relevant_hps
    }
    return prot2hps, ppi_network, prots_by_fold, hp2ancestors


def dict2matrix(data_dict, key2id, val2id, has_score=False):
    data_matrix = np.zeros((len(key2id), len(val2id)), np.float32)
    score = 1.0
    for entity, neighbour_list in data_dict.items():
        if entity not in key2id:
            continue
        for neighbour in neighbour_list:
            if has_score:
                neighbour, score = neighbour
            if neighbour not in val2id:
                continue
            data_matrix[key2id[entity], val2id[neighbour]] = score
    return data_matrix


def get_data_matrices(
    data_path,
    is_temporal,
    validate_prots_with_hpolabeler=True,
    filter_genes_with_multi_prots=True,
    filter_genes_without_valid_annotation=True,
    use_full_ppi=False,
    network_name="string",
):
    prot2hps, ppi_network, prots_by_fold, hp2ancestors = get_data_dicts(
        data_path,
        is_temporal,
        validate_prots_with_hpolabeler,
        filter_genes_with_multi_prots,
        filter_genes_without_valid_annotation,
        network_name=network_name,
    )

    # Create hpo ancestory matrix
    relevant_hps = list(hp2ancestors.keys())
    hp2id = dict(zip(relevant_hps, range(len(relevant_hps))))
    hp2ancestors_matrix = dict2matrix(hp2ancestors, hp2id, hp2id)

    # Create prot2hp matrix
    relevant_prots = list(prot2hps.keys())
    prot2id = dict(zip(relevant_prots, range(len(relevant_prots))))
    prot2hp_matrix = dict2matrix(prot2hps, prot2id, hp2id)
    # Create protids by folds
    protids_by_fold = {
        fold: np.array([prot2id[prot] for prot in prots_by_fold[fold]]) for fold in prots_by_fold
    }

    # Create ppi matrix
    if use_full_ppi:
        ppi_prots = set()
        ppi_prots.update(ppi_network.keys())  # TODO double check
        relevant_prots = list(prot2hps.keys()) + list(
            ppi_prots - prot2hps.keys()
        )
        prot2id = dict(zip(relevant_prots, range(len(relevant_prots))))
    ppi_matrix = dict2matrix(ppi_network, prot2id, prot2id, has_score=True)
    id2prot = dict(zip(prot2id.values(), prot2id.keys()))

    prot2hp_matrix = tf.cast(
        tf.not_equal(
            tf.linalg.matmul(
                prot2hp_matrix,
                hp2ancestors_matrix,
                a_is_sparse=True,
                b_is_sparse=True,
            ),
            0,
        ),
        tf.float32,
    ).numpy()

    return (
        prot2hp_matrix,
        hp2ancestors_matrix,
        ppi_matrix,
        protids_by_fold,
        id2prot,
    )

def load_dataset(dataset_dir):
    dataset_path = Path(dataset_dir)
    pickle_file_names = ["graph.p", "labels.p", "fold_ids.p"]
    data_objs = []
    for pickle_file_name in pickle_file_names:
        pickle_path = dataset_path / pickle_file_name
        with open(pickle_path, "rb") as f:
            data_objs.append(pickle.load(f))
    return tuple(data_objs)