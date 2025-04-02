#!/bin/bash


python ModalityDynMM/chestx/chestx_dynmm_2branches.py --freeze --dir test_chestx/ --reg 0.05 --noise --noise_config corruption_05 > results/vanilla_dynmm_train_reg_05_corruption_05.txt

python ModalityDynMM/chestx/chestx_dynmm_q_2branches.py --freeze --dir test_chestx/ --reg 0.05 --noise --noise_config corruption_05 > results/q_dynmm_train_reg_05_corruption_05.txt

python ModalityDynMM/chestx/chestx_dynmm_uq_2branches.py --freeze --dir test_chestx/ --reg 0.5 --noise --noise_config corruption_05 --uq_loss > results/uq_dynmm_train_reg_5_corruption_05.txt

python ModalityDynMM/chestx/chestx_dynmm_2branches.py --freeze --dir test_chestx/ --reg 0.05 --noise --noise_config swap_05 > results/vanilla_dynmm_train_reg_05_swap_05.txt

python ModalityDynMM/chestx/chestx_dynmm_q_2branches.py --freeze --dir test_chestx/ --reg 0.05 --noise --noise_config swap_05 > results/q_dynmm_train_reg_05_swap_05.txt

python ModalityDynMM/chestx/chestx_dynmm_uq_2branches.py --freeze --dir test_chestx/ --reg 0.5 --noise --noise_config swap_05 --uq_loss > results/uq_dynmm_train_reg_5_swap_05.txt

python ModalityDynMM/chestx/chestx_dynmm_2branches.py --freeze --dir test_chestx/ --reg 0.05 --noise --noise_config dropout_05 > results/vanilla_dynmm_train_reg_05_dropout_05.txt

python ModalityDynMM/chestx/chestx_dynmm_q_2branches.py --freeze --dir test_chestx/ --reg 0.05 --noise --noise_config dropout_05 > results/q_dynmm_train_reg_05_dropout_05.txt

python ModalityDynMM/chestx/chestx_dynmm_uq_2branches.py --freeze --dir test_chestx/ --reg 0.5 --noise --noise_config dropout_05 --uq_loss > results/uq_dynmm_train_reg_5_dropout_05.txt



