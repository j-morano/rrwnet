# Training RRWNet

This directory contains the code to train the RRWNet model.

Configuration options and hyperparameters can be found in `config.py`.

The data can be downloaded from the following link:
- <https://drive.google.com/drive/folders/1LHOrkaHZh0O3kOIXRVD23904ZtvVzCnV?usp=sharing>


To train the model, run the following command:

```bash
python3 train.py --dataset RITE-train --model RRWNet
```

The available datasets for training are `RITE-train` and `HRF-Karlsson-w1024`, while the available models are `RRWNet`, `RRWNetAll`, `RRUNet`, `WNet`, and `UNet`. See the paper for more details.


Training logs and weights will be saved under the `__training/` directory.
