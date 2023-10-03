# Optimal Transport for Multilingual Word Alignment

This is the github repository for research into using optimal transport for multilingual word alignment. This research was conducted by Joshua Hong under the mentorship of Yuki Arase at Osaka University

## General Information

This repository aims to explore the viability of different optimal transport techniques for multilingual word alignment. To learn more about optimal transport, click [here](https://pythonot.github.io/quickstart.html)! In the given files, one can train and evaluate models with difference types of optimal transport, including balanced, unbalanced, and partial. We build off of the techniques of past research into neural based multilingual word alignment and add these optimal transport methods on top of the methods used in AccAlign and AwesomeAlign.

We find that some optimal transport methods are competitive with SOTA results for multilingual word alignment. However, OT seems to suffer from setting transferability in the supervised setting. Further research into this topic could experiment with different formulations for fertility and distance, different base LLMs, and alternative loss functions for finetuning and training. Additional results can be found in the data spreadsheet.

## Requirements

To use this repository, the following dependencies are required: 

```
gensim
tqdm
transformers
cupy-cuda102
cython
POT
nltk
tensorboard
torch
torch_optimizer
torchmetrics
adapter-transformers
numpy
boto3
filelock
requests
tokenizers
```

## Data 

Links to the test set used in the paper are here: 

| Language Pair  |   Type |Link |
| ------------- | ------------- | ------------- |
| En-De |   Gold Alignment | www-i6.informatik.rwth-aachen.de/goldAlignment/ |
| En-Fr |   Gold Alignment | http://web.eecs.umich.edu/~mihalcea/wpt/ |
| En-Ro |   Gold Alignment | http://web.eecs.umich.edu/~mihalcea/wpt05/ |
| En-Zh |   Gold Alignment | https://nlp.csai.tsinghua.edu.cn/~ly/systems/TsinghuaAligner/TsinghuaAligner.html |
| En-Ja |   Gold Alignment | http://www.phontron.com/kftt |
| En-Sv |   Gold Alignment | https://www.ida.liu.se/divisions/hcs/nlplab/resources/ges/ |

Additionally, the training set and validation set used in the AccAlign paper can be found here [here](https://drive.google.com/file/d/19X0mhTx6-EhgILm7_mtVWrT2qal-o-uV/view?usp=share_link)

Unzip and place the folders into the data/datasets. Then run `python optimalTransport.py` to create the files needed for AccAlign and AwesomeAlign training and evaluation.

## Evaluation

To evaluate a given model with a specified dataset, run `accalign.sh` or `awesomealign.sh`. In these files, specify the source and target files for the dataset, the hyperparameters, as well as the adapter/model to be used. To find the best hyperparameters, run `hyperparam_search.sh` (make sure to adjust the file to your needs). Additionally, the files `optimal_transport/AccAlign/aligner/word_align.py` and `optimal_transport/AccAlign/self_training_modeling_adapter.py` can be modified to change the preprocessing/postprocessing steps for alignment extraction. 

Upon running these scripts, the AER, F1, recall, and precision will be printed for each layer. 

## Training

To train a model, run `train_accalign.sh` or `train_awesomealign.sh`. In these files, specify the source and target files for the training dataset and the evaluation dataset. Additionally, set the desired hyperparameters. 

## LaBSE

You can access the LaBSE model used by AccAlign [here](https://huggingface.co/sentence-transformers/LaBSE) . 

## Adapter Checkpoints 

The multilingual adapter checkpoint for AccAlign is [here](https://drive.google.com/open?id=1eB8aWd4iM6DSQWJZOA5so4rB4MCQQyQf&usp=drive_copy) . 

## References

The git repositories that this repo builds off of are [AccAlign](https://github.com/sufenlp/AccAlign/tree/main) and [AwesomeAlign](https://github.com/neulab/awesome-align)
