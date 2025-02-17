# RRWNet

This is the official repository of the paper ["RRWNet: Recursive Refinement Network for Effective Retinal Artery/Vein Segmentation and Classification"](https://doi.org/10.1016/j.eswa.2024.124970), by José Morano, Guilherme Aresta, and Hrvoje Bogunović, published in _Expert Systems with Applications_ (2024).

[[`arXiv`](https://doi.org/10.48550/arXiv.2402.03166)] [`ESWA`](https://doi.org/10.1016/j.eswa.2024.124970)] [[`BibTeX`](#citation)]


![Graphical_abstract](https://github.com/j-morano/rrwnet/assets/48717183/a573ce81-1b15-4dad-8cd7-c55bb1a049ef)


This approach builds on our previous work presented in the paper ["Simultaneous segmentation and classification of the retinal arteries and veins from color fundus images"](https://doi.org/10.1016/j.artmed.2021.102116).


## TODO

- [x] Basic inference code.
- [x] Training code.
- [x] Preprocessed training and testing data.
- [x] Data preprocessing code.
- [x] Evaluation code.
- [x] Better documentation.
- [x] Single model evaluation.
- [ ] Easier execution of the evaluation.


## Predictions and weights

The predictions for the different datasets as well as the weights for the proposed RRWNet model can be found at the following link:

- <https://drive.google.com/drive/folders/1Pz0z-OxzEft5EWGbZ3MeeqQNWqvv2ese?usp=sharing>


The model trained on the RITE dataset was trained using the original image resolution, while the model trained on HRF was trained using images resized to a width of 1024 pixels. The weights for the RITE dataset are named `rrwnet_RITE_1.pth`, while the weights for the HRF dataset are named `rrwnet_HRF_0.pth`.
Please note that the size of the images used for training is important when using the weights for predictions.


## Setting up the environment


The code was tested using Python 3.10.10.
However, it should work with other Python versions and package managers.
Just make sure to install the required packages listed in `requirements.txt`.
If you want to use this exact Python version, it can be easily installed using `pyenv` as shown in the next section.
Otherwise, you can skip to the [Requirements](#requirements) section.


### Python 3.10.10 (`pyenv`)

> [!IMPORTANT]
> The following steps are only necessary if you want to install Python 3.10.10 using `pyenv`.

Install `pyenv`.
```sh
curl https://pyenv.run | bash
```

Install `clang`. _E.g._:
```sh
sudo dnf install clang
```

Install Python version 3.10.10.
```sh
CC=clang pyenv install -v 3.10.10
```

Create and activate Python environment.
```sh
~/.pyenv/versions/3.10.10/bin/python3 -m venv venv/
source venv/bin/activate  # bash
. venv/bin/activate.fish  # fish
```

Update `pip`.

```sh
pip install --upgrade pip
```


### Requirements

Create and activate Python environment.
```sh
python -m venv venv/
source venv/bin/activate  # bash
. venv/bin/activate.fish  # fish
```

Install requirements using `requirements.txt`.

```sh
pip3 install -r requirements.txt
```


## Preprocessing (optional)

You can preprocess the images offline using the `preprocessing.py` script. The script will enhance the images and masks and save them in the specified directory.

```bash
python3 preprocessing.py --images-path data/images/ --masks-path data/masks/ --save-path data/enhanced
```


## Get predictions

To get predictions using the provided weights, run the `get_predictions.py` script. The script will save the predictions in the specified directory.
If the images were not previously preprocessed, you can use the `--preprocess` flag to preprocess the images on the fly.

```bash
python3 get_predictions.py --weights rrwnet_RITE_1.pth --images-path data/images/ --masks-path data/masks/ --save-path predictions/ --preprocess
```


## Refine existing predictions

You can refine existing predictions (e.g., from a different model) using the same `get_predictions.py` script. The script will save the refined predictions in the specified directory.
Just make sure to provide the path to the predictions and the weights to be used for the refinement.
Also, do not forget to use the `--refine` flag and do not use the `--preprocess` flag.

```bash
python3 get_predictions.py --weights rrwnet_RITE_refinement.pth --images-path data/U-Net_predictions/ --masks-path data/masks/ --save-path refined_predictions/ --refine
```


## Training

All training code can be found in the `train/` directory. The training script is `train.py`, and the configuration file, with all the hyperparameters and command line arguments, is `config.py`.
Please follow the instructions in [train/README.md](train/README.md) to train the model.
The `train/` directory also contains the code to get the predictions of the model on the test set, which are then used for the evaluation.


## Evaluation

All evaluation code can be found in the `eval/` directory.
Please follow the instructions in [eval/README.md](eval/README.md).



## Contact

If you have any questions or problems with the code or the paper, please do not hesitate to open an issue in this repository (preferred) or contact me at `jose.moranosanchez@meduniwien.ac.at`.



## Citation

If you use this code, the weights, the preprocessed data, or the predictions in your research, we would greatly appreciate it if you give a star to the repo and cite our work:

```
@article{morano2024rrwnet,
    title={RRWNet: Recursive Refinement Network for Effective Retinal Artery/Vein Segmentation and Classification},
    author={Morano, Jos{\'e} and Aresta, Guilherme and Bogunovi{\'c}, Hrvoje},
    journal={Expert Systems with Applications},
    year={2024},
    doi={10.1016/j.eswa.2024.124970}
}
```

Also, if you use any of the public datasets used in this work, please cite the corresponding papers:

- **RITE**
    + **Images**: Staal, Joes, et al. "Ridge-based vessel segmentation in color images of the retina." _IEEE transactions on medical imaging_ 23.4 (2004): 501-509.
    + **Annotations**: Hu, Qiao, Michael D. Abràmoff, and Mona K. Garvin. "Automated separation of binary overlapping trees in low-contrast color retinal images." _Medical Image Computing and Computer-Assisted Intervention–MICCAI 2013: 16th International Conference, Nagoya, Japan, September 22-26, 2013, Proceedings, Part II 16_. Springer Berlin Heidelberg, 2013.
- **LES-AV**
    + **Images and annotations**: Orlando, José Ignacio, et al. "Towards a glaucoma risk index based on simulated hemodynamics from fundus images." _Medical Image Computing and Computer Assisted Intervention–MICCAI 2018: 21st International Conference, Granada, Spain, September 16-20, 2018, Proceedings, Part II 11_. Springer International Publishing, 2018.
- **HRF**
    + **Images**: Budai, Attila, et al. "Robust vessel segmentation in fundus images." _International journal of biomedical imaging 2013.1_ (2013): 154860.
    + **Annotations**: Chen, Wenting, et al. "TW-GAN: Topology and width aware GAN for retinal artery/vein classification." _Medical Image Analysis_ 77 (2022): 102340.
