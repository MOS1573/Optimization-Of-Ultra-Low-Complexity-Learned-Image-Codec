```bash 
conda create -n  TokenCompression python=3.12
conda activate TokenCompression
conda install pip 
pip install -r requirements.txt
```



## generate_csv_dataset.py

The `generate_csv_dataset.py` script is used to generate a CSV dataset from a given input. This script takes an input file and processes it to produce a CSV file. 

#### Usage

Edit the `dataset` path and and `dataset_csv` in the `configs/general.yaml` file.
To use the `generate_csv_dataset.py` script, run the following command:

```bash
python generate_csv_dataset.py 
```

It will generate a CSV file at `data/dataset_csv.csv`.
