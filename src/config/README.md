# Content
This folder provides a configuration for running the SciBERT-CLS-modal model on the standard train/test split of MiST.
Configurations are stored as JSON files.
To enable re-use of configuration parts without duplication, configurations are organized in a modular way:
* [cross_validation](cross_validation): Contains paths to train and validation files for each split in cross-validated training.
* [evaluation](evaluation): Configs for training and evaluating models in cross-validated training, i.e., training and evaluating a model on each of the splits. Can be used as an argument to [evaluate.py](../evaluate.py).
* [model](model): Configs for training and evaluating **a single** model on a given train-val-test split. 
* [task](task): Configs for tasks, e.g., classification on MiST. These configs are referred to in the ``tasks`` attribute of the model configs. Note that the identifier used for a specific task in ``tasks`` and ``heads`` of a model config (the latter for specifying task-specific output heads) and in cross-validation configs must be identical (e.g., ``mist``).


We also provide an example configuration for multi-task training with EPOS ([Marasovic et al. (2016)](https://aclanthology.org/2016.lilt-14.3/). 
In order to run this configuration, you need to get the corpus and convert it to the following format:
```
SINGLE-LABEL

#<sentence ID>
<target modal verb, e.g., "can">
<position of target modal verb, 0-indexed>
<label of target modal verb, e.g., "dy">
<token_1>
<token_2>
...
<token_n>

#<sentence ID>
<target modal verb, e.g., "can">
<position of target modal verb, 0-indexed>
<label of target modal verb, e.g., "dy">
<token_1>
<token_2>
...
<token_m>
```