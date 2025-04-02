python ModalityDynMM/chestx/chestx_dynmm_2branches.py --freeze --dir test_chestx/ --reg 0.05 --noise --noise_config gaussian_50 > results/vanilla_dynmm_train_reg_05_gaussian_50.txt

python ModalityDynMM/chestx/chestx_dynmm_q_2branches.py --freeze --dir test_chestx/ --reg 0.05 --noise --noise_config gaussian_50 > results/q_dynmm_train_reg_05_gaussian_50.txt

python ModalityDynMM/chestx/chestx_dynmm_uq_2branches.py --freeze --dir test_chestx/ --reg 0.5 --noise --noise_config gaussian_50 --uq_loss > results/uq_dynmm_train_reg_5_gaussian_50.txt

python ModalityDynMM/chestx/chestx_dynmm_2branches.py --freeze --dir test_chestx/ --reg 0.05 --noise --noise_config blur_3 > results/vanilla_dynmm_train_reg_05_blur_3.txt

python ModalityDynMM/chestx/chestx_dynmm_q_2branches.py --freeze --dir test_chestx/ --reg 0.05 --noise --noise_config blur_3 > results/q_dynmm_train_reg_05_blur_3.txt

python ModalityDynMM/chestx/chestx_dynmm_uq_2branches.py --freeze --dir test_chestx/ --reg 0.5 --noise --noise_config blur_3 --uq_loss > results/uq_dynmm_train_reg_5_blur_3.txt

python ModalityDynMM/chestx/chestx_dynmm_2branches.py --freeze --dir test_chestx/ --reg 0.05 --noise --noise_config mask_01_2 > results/vanilla_dynmm_train_reg_05_mask_01_2.txt

python ModalityDynMM/chestx/chestx_dynmm_q_2branches.py --freeze --dir test_chestx/ --reg 0.05 --noise --noise_config mask_01_2 > results/q_dynmm_train_reg_05_mask_01_2.txt

python ModalityDynMM/chestx/chestx_dynmm_uq_2branches.py --freeze --dir test_chestx/ --reg 0.5 --noise --noise_config mask_01_2 --uq_loss > results/uq_dynmm_train_reg_5_mask_01_2.txt


