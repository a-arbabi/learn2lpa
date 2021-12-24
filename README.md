# Learning to Propagate Labels to Predict Phenotype Associations for Novel Genes

## Data preparation
The preprocessed data used in our experiments is available in this repository located at `data/preprocessed_data.tar.gz`. You can extract and use this dataset as follows:
```
mkdir -p /path/to/preprocessed_dataset/
tar -zxf data/preprocessed_data.tar.gz -C /path/to/preprocessed_dataset/
```

Alternatively, you can repeat the data preprocssing steps by first downloading the raw data and then running the preprocessing code:
```
python data_tools/data_download.py /path/to/raw_dataset/
python data_tools/create_datasets.py /path/to/raw_dataset/ /path/to/preprocessed_dataset/ 
```

The preprocessed datasets are stored in two subdirectories, corresponding to cross-validation and temporal experiments: 
```
/path/to/preprocessed_dataset/ppi_hpo_string_cv/
/path/to/preprocessed_dataset/ppi_hpo_string_temporal/
```

## Experiments
Run a cross-validation (`experiment_type=cv`) or temporal (`experiment_type=temporal`) experiment using the following script. Notice that there are two different dataset directories created for cv and temporal experiments.
```
python run_main_experiment.py /path/to/preprocessed_dataset/ppi_hpo_string_"$experiment_type"/ $experiment_type
```


