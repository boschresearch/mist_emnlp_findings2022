<!---

    Copyright (c) 2022 Robert Bosch GmbH and its subsidiaries.

-->

# Modals in Scientific Text (MiST)
This repository contains the companion material for the following publication:

> Sophie Henning, Nicole Macher, Stefan Grünewald, and Annemarie Friedrich. MiST: a Large-Scale Annotated Resource and Neural Models for Functions of Modal Verbs in English Scientific Text. In Findings of the Association for Computational Linguistics: EMNLP 2022. Abu Dhabi.

Please cite this paper if using the dataset or the code, and direct any questions regarding the dataset and code at [Sophie Henning](mailto:sophieelisabeth.henning@de.bosch.com). The paper can be found [here](https://aclanthology.org/2022.findings-emnlp.94/).

## Purpose of this Software 

This software is a research prototype, solely developed for and published as part of the publication cited above. It will neither be maintained nor monitored in any way.

## Data
We release **Modals in Scientific Text** (MiST), a dataset to foster research on functions of modal verbs in English scientific text. In scientific writing, modal verbs are a popular lingustic device to hedge statements, but they are also used, e.g., to denote capabilities. Hence, decoding their function is important, e.g., for extracting information from scientific text.

MiST covers papers from five scientific domains (computational linguistics, materials science, agriculture, computer science, and earth science) and contains 3,737 manually annotated modal verb instances.

For more details on the annotation scheme and dataset construction, please refer to the paper.
Further information on how the data is stored can be found in [data/README.md](data/README.md).

## Code
We provide code for training and evaluating $SB_{CLS,modal}$ models (see paper) on MiST. We also provide code for multi-task training with additional corpora.

### Requirements
The code requires the following dependencies:
* [Python](https://www.python.org/) == 3.8.12
* [Huggingface Transformers](https://github.com/huggingface/transformers) == 4.5.0
* [MLFlow](https://mlflow.org/) ==1.16.0
* [torchmetrics](https://torchmetrics.readthedocs.io/en/latest/) == 0.3.1
* [NumPy](https://numpy.org/) == 1.21.2


You can install all the above dependencies easily using [Conda](https://docs.conda.io/en/latest/)
and the ```environment.yml``` file provided by us:
```bash
conda env create -f environment.yml
conda activate stepsenv
```

### Training and Evaluating Models
To train and evaluate $SB_{CLS,modal}$ models as described in the paper (5 models based on 5 different splits of the training set into a train and a validation set), run `python src/evaluate.py --repo_path [REPO_PATH] src/config/evaluation/scibert-cls-modal.json`.

Note that this evaluation configuration assumes that you have stored the [SciBERT model](https://huggingface.co/allenai/scibert_scivocab_uncased) in the `pretrained_embeddings` folder.

For performing multi-task training with EPOS ([Marasovic et al. (2016)](https://aclanthology.org/2016.lilt-14.3/), you need to get the corpus and convert it to the format specified in [src/config/README.md](src/config/README.md).
You can then run the training using `python src/evaluate.py --repo_path [REPO_PATH] src/config/evaluation/scibert-cls-modal_with-EPOS.json`.

## License
The code in this repository is open-sourced under the AGPL-3.0 license. See the [LICENSE](LICENSE.txt) file for details. For a list of other open source components included in this project, see the file [3rd-party-licenses.txt](3rd-party-licenses.txt).

The MiST annotations located in [data](data) are licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/) (CC BY 4.0). For licenses of the underlying papers, please refer to [data/metadata.csv](data/metadata.csv).