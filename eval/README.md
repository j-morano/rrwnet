# Evaluation

In this directory, there are multiple files that are used to evaluate the performance of the models from the predictions.

The ground truths for the evaluations, the predictions of our model as well as the other SOTA models can be found in the following link:
- <https://drive.google.com/drive/folders/1E2TIjwqMb-YM2IuTSPE5L4NeZ-Q04ktU?usp=sharing>

The data should be placed in the `_Evaluation_Data/` directory under `eval/`.


There are two types of evaluations that can be done:
1. Single model evaluation, where the predictions of a single model are evaluated.
2. State-of-the-art (SOTA) evaluation, where the predictions of multiple models are evaluated.


## Single model evaluation

Once the predictions are generated (see [train/README.md](../train/README.md)), they can be evaluated using the following commands.

```bash
# Activate the virtual environment
source ../venv/bin/activate
# Evaluate the model
compare_results.py -p <path_to_the_predictions> -d <dataset_name>
```

This command will evaluate the predictions of the model on the specified dataset. The results will be saved in the `__results/` directory in a JSON file with the name of the dataset.


## SOTA evaluation


To evaluate the model and the baselines, copy the predictions of the models to the corresponding dataset directory in the `_Evaluation_Data/` directory, naming the subdirectory with the name of the model. Then, add the subpath of the predictions to the list of dataset paths in the `compare_results.py` file. For example, for the predictions of a model named `rrwnet_version2` for the `RITE-test` dataset, the path should be added as follows:

```python
    # line 35 of compare_results.py
    if args.predictions_path is None:
        dataset.paths = dataset.paths.union({
            'gt2_qureshi',
            'rrwnet',
            'unet',
            'unet/refined_AV',
            'chen',
            'chen/refined_AV',
            'karlsson',
            'karlsson/refined_AV',
            'galdran',
            'galdran/refined_AV',
            'rrwnet_version2'  # > ADDED LINE
        })
```

Then, run the following command to evaluate the predictions of the models.

```bash
# Activate the virtual environment, if not already activated
source ../venv/bin/activate
# Run the SOTA evaluation, e.g., for the RITE-test dataset
python3 compare_results.py -d RITE-test
```

As before, the results will be saved in the `__results/` directory in a JSON file with the name of the dataset.


Please check the [README.md](../README.md) file in the root directory for the citation information of the different models.