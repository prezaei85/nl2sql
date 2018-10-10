# Natural Language Database Queries
This repository explores translation of natural language questions to SQL code to get data from relational databases. A lot of the code is inspired or taken directly from the [Coarse2Fine](https://github.com/donglixp/coarse2fine) repo, which is the model used here.

This repo provides: 
- inference files for running the [Coarse2Fine](https://github.com/donglixp/coarse2fine) model with **new input questions** over tables from [WikiSQL](https://github.com/salesforce/WikiSQL),
- a sample Flask app that uses the inference files to serve the model, and 
- a simplified implementation of [execution guidance](https://arxiv.org/abs/1807.03100) when decoding the SQL code to improve the accuracy of the model. 

You need:
- **[Stanford CreNLP](https://github.com/stanfordnlp/python-stanford-corenlp)** for data annotation 
- **[Spacy](https://spacy.io/usage/linguistic-features#pos-tagging)** for part-of-speech tagging of the natural language question
- **[PyTorch 0.2.0.post3](http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl)** from [previous versions](https://pytorch.org/get-started/previous-versions/) of PyTorch (using GPU)

## Install dependencies
```
pip install -r requirements.txt
```
## Download pretrained model
Download pretrained model from [here](https://drive.google.com/file/d/18oMNo4yC01gwMjHcfmE-_G5qE7X5SLYt/view?usp=sharing) and unzip it in the folder ```pretrained``` in the root folder. 

## Infer model 
You can modify and run the following python script for infering the model with new input questions:
```
python run_model.py
```
## Use execution guidance for evaluating the model over test set
Evaluate the model over all questions from tables of the test set in WikiSQL:
```
cd src/
python evaluate.py -model_path ../pretrained/pretrain.pt -beam_search
```

