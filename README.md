# Quality-and-Uncertainty-Aware-Dynamic-Fusion
We extend the [DynMM](https://github.com/zihuixue/DynMM) repository the with quality assessor and uncertainty estimator modules and quality and uncertainty-aware loss function in order to improve robustness against poor-quality or corrupt data.

## Setup:
To reproduce our results:

(1) Clone the [MultiBench](https://github.com/pliang279/MultiBench) library and setup the virtual environment in the instructions.

(2) Clone the DynMM repository.

(3) Add the files in Quality and Uncertainty-Aware Dynamic Fusion repository to the ModalityLevel folder of DynMM.

(4) Add the modified ModalityLevel folder to the MultiBench folder.

(5) Clone the [mm-health-bench](https://github.com/konst-int-i/mm-health-bench) repository into the MultiBench folder.

(6) Download the ChestX dataset following the mm-health-bench instructions.

## Training experts:
```
# train text-only expert
python chestx_uni.py --mod 0

#train image-text late fusion expert
python chestx_mm.py --fuse 1

```

## Training dynamic networks on (noisy) data:
```
# For vanilla DynMM
python chestx_dynmm_2branches.py (--freeze) (--noise) (--noise_config <predefined noise.py congiguration>) --reg <lambda/resource allocation parameter>

# For DynMM + Quality Assessors
python chestx_dynmm_q_2branches.py (--freeze) (--noise) (--noise_config <predefined noise.py congiguration>) --reg <lambda/resource allocation parameter>

# For DynMM  + Quality Assessors + Uncertainty Estimators + quality and uncertainty-aware function
python chestx_dynmm_uq_2branches.py (--freeze) (--noise) (--noise_config <predefined noise.py congiguration>) --reg <lambda/resource allocation parameter> (--uq_loss)
```

The models are tested at the end with test data of the same corruption type using a held-out test dataset.
All models are saved in ./logs/chestx

# Replicating our experiments
To replicate our experiments, simply run:
```
sh run_text_corruption_tests.sh
sh run_image_corruption_tests.sh
```
# Helper files

The functions that add noise and load noisy ChestX data are available in ```noise.py```.

```chestx_count_flop.py``` outputs the FLOPs count / 2.

The quality and uncertainty-aware loss function is defined in ```train_uq_loss.py```.

```chestx_utils.py``` contains helper functions to load the clean ChestX dataset, analyse class imbalance and define the ```pos_weight``` vector for the Pytorch ```BCEWithLogitsLoss``` criterion to handle class imbalance.

The results from our experiments are available in ```results/```.

