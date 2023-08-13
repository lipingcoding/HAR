# HAR

## Paper data and code

This is the code for the TKDE submission paper: Stage-Aware Hierarchical Attentive Relational Network for Diagnosis Prediction. We have implemented our methods in **Pytorch** and **DGL**.
Here, we provide the source code for model-variant HAR-LSTM. 

## Usage

You need to download MIMIC-III and MIMIC-IV datasets by yourself. Then you need finish data-processing as following:
```
cd data_util;
python simplify_semmed.py
python prepare_icd2cui.py
python filter_semmed_kg_edges.py 
python process_mimic.py
```


Then you can run the file `main.py` to train the model.



```

## Requirements

- Python 3
- PyTorch
- DGL
