import requests
from pathlib import Path
import io
import gzip
import zipfile
from data_tools.data_utils import file2dict
import argparse


def download_ensembl2prot(strings, output_path="ensembl2prot.txt"):
    url = "https://www.uniprot.org/uploadlists/"
    chunk_size = 200
    with open(output_path, "w") as f:
        f.write("ensembl\tprot\n")
        for i in range(0, len(strings), chunk_size):
            if i % 1000 == 0:
                print(i / len(strings))
            chunk = strings[i : i + chunk_size]
            params = {
                "from": "ENSEMBL_ID",
                "to": "SWISSPROT",
                "format": "tab",
                "query": " ".join(chunk),
            }
            content = requests.get(url, params=params).text
            content = "\n".join(content.splitlines()[1:]) + "\n"
            f.write(content)


def download_string2prot(strings, output_path="string2prot.txt"):
    url = "https://www.uniprot.org/uploadlists/"
    chunk_size = 200
    with open(output_path, "w") as f:
        f.write("string\tprot\n")
        for i in range(0, len(strings), chunk_size):
            if i % 1000 == 0:
                print(i / len(strings))
            chunk = strings[i : i + chunk_size]
            params = {
                "from": "STRING_ID",
                "to": "SWISSPROT",
                "format": "tab",
                "query": " ".join(chunk),
            }
            content = requests.get(url, params=params).text
            content = "\n".join(content.splitlines()[1:]) + "\n"
            f.write(content)


def download_gene2uniprot(genes, output_path="gene2prot.txt"):
    url = "https://www.uniprot.org/uniprot/"
    with open(output_path, "w") as f:
        f.write("entrez_gene\tgene_name\tuniprot\n")
        for i, gene in enumerate(genes):
            if i % 100 == 0:
                print(i / len(genes))
            params = {
                "format": "tab",
                "query": f"geneid:{gene[0]}+AND+gene_exact:{gene[1]}+AND+reviewed:yes+AND+organism:9606",
            }
            content = requests.get(url, params=params).text
            for prot in [row.split()[0].strip() for row in content.splitlines()[1:]]:
                f.write(f"{gene[0]}\t{gene[1]}\t{prot}\n")


def download_all_files(output_dir):
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    hpo_url_2017_02_24 = "https://data.bioontology.org/ontologies/HP/submissions/562/download?apikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb"
    hpo_url_2018_07_25 = "https://data.bioontology.org/ontologies/HP/submissions/572/download?apikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb"
    string_ppi_url = "https://version-10-5.string-db.org/download/protein.links.v10.5/9606.protein.links.v10.5.txt.gz"
    #genemania_ppi_url = "http://genemania.org/data/current/Homo_sapiens.COMBINED/COMBINED.DEFAULT_NETWORKS.BP_COMBINING.txt"
    genemania_ppi_url = "http://genemania.org/data/archive/2017-03-12/Homo_sapiens.COMBINED/COMBINED.DEFAULT_NETWORKS.BP_COMBINING.txt"
    hpolabeler_url = "https://ndownloader.figshare.com/files/21747147"

    # hp_2017_02_24.obo
    with open(base_dir / "hp_2017_02_24.obo", "w") as f:
        response = requests.get(hpo_url_2017_02_24)
        f.write(response.text.replace("; http", ", http"))

    # hp_2018_07_25.obo
    with open(base_dir / "hp_2018_07_25.obo", "w") as f:
        response = requests.get(hpo_url_2018_07_25)
        f.write(
            "\n".join(
                [
                    line
                    for line in response.text.replace("; http", ", http").splitlines()
                    if not line.startswith("def:") and not line.startswith("xref:")
                ]
            )
        )

    # 9606.protein.links.v10.5.txt
    with open(base_dir / "9606.protein.links.v10.5.txt", "wb") as f:
        response = requests.get(string_ppi_url)
        source_handle = io.BytesIO(response.content)
        with gzip.GzipFile(fileobj=source_handle) as f_source:
            f.write(f_source.read())

    # COMBINED.DEFAULT_NETWORKS.BP_COMBINING.txt
    with open(base_dir / "COMBINED.DEFAULT_NETWORKS.BP_COMBINING.txt", "w") as f:
        response = requests.get(genemania_ppi_url)
        for row in response.text.splitlines():
            cells = row.split()
            f.write(row.strip()+'\n')
            if cells[0] == 'Gene_A':
                continue
            f.write(f"{cells[1]}\t{cells[0]}\t{cells[2]}\n")

    # HPOLabeler_data/
    response = requests.get(hpolabeler_url)
    source_handle = io.BytesIO(response.content)
    with zipfile.ZipFile(source_handle, "r") as zip_ref:
        zip_ref.extractall(base_dir)

    # gene2prot.txt
    temporal_files = {
        "train": base_dir
        / "HPOLabeler_data/annotation/ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype_2017_02_24.txt",
        "valid": base_dir
        / "HPOLabeler_data/annotation/ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype_2018_03_09.txt",
        "test": base_dir
        / "HPOLabeler_data/annotation/ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype_2018_12_21.txt",
    }
    all_genes = {
        gene
        for gene2hp_file in temporal_files.values()
        for gene in file2dict(gene2hp_file, (0, 1), 3)
    }
    download_gene2uniprot(list(all_genes), output_path=base_dir / "gene2prot.txt")

    # string2prot.txt
    ppi_string = file2dict(
        base_dir / "9606.protein.links.v10.5.txt",
        0,
        (1, 2),
        separator=" ",
    )
    download_string2prot(
        list(ppi_string.keys()), output_path=base_dir / "string2prot.txt"
    )

    # ensmbl2prot.txt
    ppi_genemania = file2dict(
        base_dir / "COMBINED.DEFAULT_NETWORKS.BP_COMBINING.txt",
        0,
        (1, 2),
        separator="\t",
    )
    download_ensembl2prot(
        list(ppi_genemania.keys()), output_path=base_dir / "ensembl2prot.txt"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Download data for protein function prediction."
    )
    parser.add_argument("output_dir", help="Path to the output directory.")
    args = parser.parse_args()
    download_all_files(args.output_dir)


if __name__ == "__main__":
    main()