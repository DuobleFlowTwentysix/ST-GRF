# ST-GRF

Rainfall predicting and rainfall pattern homogeneity quantifying via spatiotemporal graph neural network

## Dataset

-Download dataset from [Baiduyun](https://pan.baidu.com/s/1foHoT88HPod38OGjsvbKoA) with code `pvcd`.

## Requirements

```
Python 3.7.0
```

```bash
pip install -r requirements.txt
```

## Experiment Setup

open `config.yaml`.
- set data path after your server name.
My path under windows operating system is:

```python
filepath:
  GPU-Server:
    knowair_fp: D:\code\my_code\ST-GRF-main\data\fanghao\rainfall\stgrf\chainair2010_2019_a.npy
    results_dir: D:\code\my_code\ST-GRF-main\data\fanghao\rainfall\stgrf\results

```

- Choose the model you want to run.

```python
#  model: MLP
#  model: LSTM
#  model: GRU
#  model: GC_LSTM
#  model: nodesFC_GRU
  model: STGRF_GRU
#  model: STGRF_GRU_nosub
#  model: STGRF_LSTM
#  model: STGRF_BILSTM
#  model: STGRF_BIGRU
#  model: STGRF_BILSTM_nosub
```

- Choose the sub-datast number in [1,2,3].

```python
 dataset_num: 3
```

- You can customize the training step size according to your preferences. The step size used in the experiments in the paper is [1,2,3,5,7]

## Run

```bash
python train.py --model 'STGRF_GRU' --pred_len '2' --dataset_num '3'
```

