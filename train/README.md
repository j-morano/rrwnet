# Training RRWNet

This directory contains the code to train the RRWNet model.

Configuration options and hyperparameters can be found in `config.py`.

The data can be downloaded from the following link:
- GitHub: <https://github.com/j-morano/rrwnet/releases/download/preds-n-data/_Data.zip>

Place the data in the `_Data/` directory under `train/`.


To train the model, run the following commands:

```bash
# Activate the virtual environment
source ../venv/bin/activate
# Train the model
python3 train.py --dataset RITE-train --model RRWNet
```

The available datasets for training are `RITE-train` and `HRF-Karlsson-w1024`, while the available models are `RRWNet`, `RRWNetAll`, `RRUNet`, `WNet`, and `UNet`. See the [paper](https://arxiv.org/pdf/2402.03166) for more details.


Training logs and weights will be saved under the `__training/` directory.


Once the model is trained, the predictions can be generated using the following command.
<!-- The `-p` flag should point to the directory containing the trained model weights, while the `-i` flag should point to the directory containing the images to be predicted. -->

```bash
python3 get_predictions.py -p <path_to_the_trained_model> -i <path_to_the_images>
```

The predictions will be saved under the `tests_predictions/` directory in the path specified by the `-p` flag.


To evaluate the predictions, see [eval/README.md](../eval/README.md).
