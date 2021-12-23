# Learning to Propagate Labels to Predict Phenotype Associations for Novel Genes

## Data preparation
First download the raw data and store in `/path/to/raw_dataset/`:
```
python data_tools/data_download.py /path/to/raw_dataset/
```

Next run the preprocessng script and store the output in `/path/to/preprocessed_dataset/`:
```
python data_tools/create_datasets.py /path/to/raw_dataset/ /path/to/preprocessed_dataset/ 
```

The preprocessed datasets are stored in the following two subdirectories, corresponding to cross-validation and temporal experiments: 
```
/path/to/preprocessed_dataset/ppi_hpo_string_cv/
/path/to/preprocessed_dataset/ppi_hpo_string_temporal/
```

## Experiments
Run a cross-validation (`experiment_type=cv`) or temporal (`experiment_type=temporal`) experiment using the following script. Notice that there are two different dataset directories created for cv and temporal experiments.
```
python run_main_experiments.py /path/to/preprocessed_dataset/ppi_hpo_string_"$experiment_type"/ $experiment_type
```


