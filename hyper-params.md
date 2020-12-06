# Hyper-parameter Tuning

This file contains the hyper-parameters we test and report in our paper (mapping to the parameters in Justfile commands).

**NOTE:** Different hardwares may have different results, for example the max q-error on Census dataset of Naru (the same hyper-parameter and python & library version) on CPU machine is 66.0, P100 GPU machine (ComputeCanada Cedar) is 57.0 and K80 GPU machine (AWS p2.xlarge) is 58.0. The result we report for neural network methods are trained and tested on P100 GPU machine and others are on CPU.

## Static Environment

### Naru

CMD: `train-naru` and `test-naru`

Model Architectures:
* Census
  * layers: 5, hc_hiddens: 16, embed_size: 8
  * layers: 4, hc_hiddens: 16, embed_size: 8
  * layers: 5, hc_hiddens: 32, embed_size: 4
  * layers: 4, hc_hiddens: 32, embed_size: 4
* Forest
  * layers: 5, hc_hiddens: 32, embed_size: 8
  * layers: 4, hc_hiddens: 64, embed_size: 8
  * layers: 5, hc_hiddens: 64, embed_size: 4
  * layers: 4, hc_hiddens: 64, embed_size: 4
* Power
  * layers: 5, hc_hiddens: 64, embed_size: 32
  * layers: 4, hc_hiddens: 64, embed_size: 32
  * layers: 5, hc_hiddens: 128, embed_size: 16
  * layers: 4, hc_hiddens: 128, embed_size: 16
* DMV
  * layers: 5, hc_hiddens: 256, embed_size: 128
  * layers: 4, hc_hiddens: 512, embed_size: 128
  * layers: 5, hc_hiddens: 512, embed_size: 64
  * layers: 4, hc_hiddens: 512, embed_size: 64

Others:
* warmups: 0, 4000, 8000
* epochs: 100
* psample: 2000
* we use natral order for all the dataset

Selected Models:

```bash
# census
just train-naru census13 original 4 16 8 embed embed True 0 0 100 base 123
just test-naru original-resmade_hid16,16,16,16_emb8_ep100_embedInembedOut_warm0-123 2000 census13 original base 123

# forest
just train-naru forest10 original 4 64 8 embed embed True 4000 0 100 base 123
just test-naru original-resmade_hid64,64,64,64_emb8_ep100_embedInembedOut_warm4000-123 2000 forest10 original base 123

# power
just train-naru power7 original 5 128 16 embed embed True 4000 0 100 base 123
just test-naru original-resmade_hid128,128,128,128,128_emb16_ep100_embedInembedOut_warm4000-123 2000 power7 original base 123

# dmv
just train-naru dmv11 original 4 512 128 embed embed True 4000 0 100 base 123
just test-naru original-resmade_hid512,512,512,512_emb128_ep100_embedInembedOut_warm4000-123 2000 dmv11 original base 123
```

### MSCN

CMD: `train-mscn` and `test-mscn`

Model Architectures:
* Census
  * num_samples: 200, hid_units: 32
  * num_samples: 400, hid_units: 16
  * num_samples: 500, hid_units: 8
  * num_samples: 600, hid_units: 4
* Forest
  * num_samples: 1000, hid_units: 64
  * num_samples: 3000, hid_units: 32
  * num_samples: 4000, hid_units: 16
  * num_samples: 5000, hid_units: 8
* Power
  * num_samples: 1000, hid_units: 128
  * num_samples: 5000, hid_units: 64
  * num_samples: 9000, hid_units: 32
  * num_samples: 10000, hid_units: 16
* DMV
  * num_samples: 1000, hid_units: 512
  * num_samples: 5000, hid_units: 512
  * num_samples: 8000, hid_units: 256
  * num_samples: 10000, hid_units: 256

Others:
* bs: 256, 512, 1024, 2048
* epochs: 100

Selected Models:

```bash
# census
just train-mscn census13 original base 500 8 100 256 100000 0 123
just test-mscn original_base-mscn_hid8_sample500_ep100_bs256_100k-123 census13 original base 123

# forest
just train-mscn forest10 original base 3000 32 100 256 100000 0 123
just test-mscn original_base-mscn_hid32_sample3000_ep100_bs256_100k-123 forest10 original base 123

# power
just train-mscn power7 original base 5000 64 100 256 100000 0 123
just test-mscn original_base-mscn_hid64_sample5000_ep100_bs256_100k-123 power7 original base 123

# dmv
just train-mscn dmv11 original base 10000 256 100 256 100000 0 123
just test-mscn original_base-mscn_hid256_sample10000_ep100_bs256_100k-123 dmv11 original base 123
```

### LW-NN

CMD: `train-lw-nn` and `test-lw-nn`

Model Architectures:
* Census
  * hid_units: 64_64_64_64
  * hid_units: 128_64_32_16
  * hid_units: 64_64_64
  * hid_units: 128_64_32
* Forest
  * hid_units: 512_256
  * hid_units: 256_256_256
  * hid_units: 256_256_128_128
  * hid_units: 256_256_128_64
* Power
  * hid_units: 512_512
  * hid_units: 512_256_128_64
  * hid_units: 256_256_256_256
  * hid_units: 512_512_256
* DMV
  * hid_units: 2048_1024_512_256
  * hid_units: 1024_1024_1024_1024
  * hid_units: 2048_1024_1024
  * hid_units: 1024_1024_1024

Others:
* bs: 32, 128, 512
* bins: 200
* epochs: 500

Selected Models:

```bash
# census
just train-lw-nn census13 original base 64_64_64 200 100000 128 0 123
just test-lw-nn original_base-lwnn_hid64_64_64_bin200_ep500_bs128_100k-123 census13 original base True 123

# forest
just train-lw-nn forest10 original base 256_256_128_64 200 100000 32 0 123
just test-lw-nn original_base-lwnn_hid256_256_128_64_bin200_ep500_bs32_100k-123 forest10 original base True 123

# power
just train-lw-nn power7 original base 512_512_256 200 100000 128 0 123
just test-lw-nn original_base-lwnn_hid512_512_256_bin200_ep500_bs128_100k-123 power7 original base True 123

# dmv
just train-lw-nn dmv11 original base 2048_1024_512_256 200 100000 32 0 123
just test-lw-nn original_base-lwnn_hid2048_1024_512_256_bin200_ep500_bs32_100k-123 dmv11 original base True 123
```

### LW-XGB

CMD: `train-lw-tree` and `test-lw-tree`

trees:
* Census: 16, 32, 64
* Forest: 128, 256, 512
* Power: 256, 512, 1024
* DMV: 2048, 4096, 8192

Selected Models:

```bash
# census
just train-lw-tree census13 original base 64 200 100000 0 123
just test-lw-tree original_base-lwxgb_tr64_bin200_100k-123 census13 original base True 123

# forest
just train-lw-tree forest10 original base 512 200 100000 0 123
just test-lw-tree original_base-lwxgb_tr512_bin200_100k-123 forest10 original base True 123

# power
just train-lw-tree power7 original base 256 200 100000 0 123
just test-lw-tree original_base-lwxgb_tr256_bin200_100k-123 power7 original base True 123

# dmv
just train-lw-tree dmv11 original base 8192 200 100000 0 123
just test-lw-tree original_base-lwxgb_tr8192_bin200_100k-123 dmv11 original base True 123
```  

### DeepDB

CMD: `train-deepdb` and `test-deepdb`

Grid Search:
* rdc_threshold: 0.2, 0.3, 0.4
* ratio_min_instance_slice: 0.001, 0.005, 0.01, 0.05
* hdf_sample_size: 1M, 10M

Selected Models:

```bash
# census
just train-deepdb census13 original 1000000 0.4 0.01 0 base 123
just test-deepdb original-spn_sample48842_rdc0.4_ms0.01-123 census13 original base 123

# forest
just train-deepdb forest10 original 1000000 0.4 0.005 0 base 123
just test-deepdb original-spn_sample581012_rdc0.4_ms0.005-123 forest10 original base 123

# power
just train-deepdb power7 original 10000000 0.3 0.001 0 base 123
just test-deepdb original-spn_sample2075259_rdc0.3_ms0.001-123 power7 original base 123

# dmv
just train-deepdb dmv11 original 1000000 0.2 0.001 0 base 123
just test-deepdb original-spn_sample1000000_rdc0.2_ms0.001-123 dmv11 original base 123
```

## Micro-Bencmark

### Preparation

#### Data Generation

CMD: `data-gen`
* skew: 0.0, 0.2, 0.4, ..., 1.8, 2.0
* corr: 0.0, 0.1, 0.2, ..., 0.9, 1.0
* dom: 10, 100, 1000, 10000

#### Workload Generation

CMD: `wkld-gen-vood` and `wkld-label`

Example: generate dataset and workload for dataset versions with 1000 domain values

```bash
# 1. generate versions
for c in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    for s in 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0; do
        just data-gen $s $c 1000
    done
done

# 2. generate queryset (can use any version to generate this workload since we use independent center values and the domains are the same)
wkld-gen-vood dom1000 skew0.0_corr0.0

# 3. generate labels for each version
for c in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    for s in 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0; do
        just wkld-label dom1000 skew${s}_corr${c} vood
    done
done
```

### Model Parameters

In this experiment, we train and test following models on every synthetic dataset (using `vood` workload) we generated. **Architecture used to report in paper**

#### Naru

CMD: `train-naru` and `test-naru`

Model Architectures:
* dom10 (domain size = 10)
  * layers: 5, hc_hiddens: 64, embed_size: 128
  * **layers: 4, hc_hiddens: 64, embed_size: 64**
  * layers: 5, hc_hiddens: 64, embed_size: 32
* dom100
  * layers: 4, hc_hiddens: 32, embed_size: 128
  * layers: 5, hc_hiddens: 32, embed_size: 64
  * **layers: 5, hc_hiddens: 32, embed_size: 32**
* dom1000
  * layers: 5, hc_hiddens: 16, embed_size: 16
  * layers: 5, hc_hiddens: 64, embed_size: 8
  * **layers: 4, hc_hiddens: 16, embed_size: 16**
* dom10000
  * layers: 3, hc_hiddens: 64, embed_size: 2
  * **layers: 4, hc_hiddens: 32, embed_size: 2**
  * layers: 5, hc_hiddens: 32, embed_size: 2

Others:
* warmups: 0
* epochs: 100

#### MSCN

CMD: `train-mscn` and `test-mscn`

Model Architectures:
* **num_samples: 1000, hid_units: 32**
* num_samples: 3000, hid_units: 8
* num_samples: 5000, hid_units: 4

Others:
* bs: 1024
* epochs: 100
* train_num: 100000

#### LW-NN

CMD: `train-lw-nn` and `test-lw-nn`

Model Architectures:
* hid_units: 256_128_64
* hid_units: 128_128_128
* **hid_units: 256_128_64_32**

Others:
* bs: 32
* bins: 200
* epochs: 500
* train_num: 100000

#### LW-Tree

CMD: `train-lw-tree` and `test-lw-tree`

* trees: 128
* bins: 200
* train_num: 100000

#### DeepDB

CMD: `train-deepdb` and `test-deepdb`

* hdf_sample_size: 1000000
* rdc_threshold: 0.3
* ratio_min_instance_slice: 0.01
