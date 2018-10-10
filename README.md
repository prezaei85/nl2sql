# Natural Language Database Queries
This repository explores translation of natural language questions to SQL code to get data from relational databases. The base model and algorithm was inspired and based upon the [Coarse2Fine](https://github.com/donglixp/coarse2fine) repo. 

This repo provides: 
- inference files for running the [Coarse2Fine](https://github.com/donglixp/coarse2fine) model with **new input questions** over tables from [WikiSQL](https://github.com/salesforce/WikiSQL),
- a sample Flask app that uses the inference files to serve the model, and 
- a simplified implementation of [execution guidance](https://arxiv.org/abs/1807.03100) when decoding the SQL code to improve the accuracy of the model. 

Here are some [slides](https://drive.google.com/open?id=10j0kv4BkVQe18fimvAgdgiU6gJe9bsMmYzCPx8Zq_ZI) for the presentation of this repo, and the Flask app page serving the model (**www.nlp2sql.com**): 

![Alt text](/flaskapp/static/app_page.png?raw=true "Flask app page")

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

## Train model
Use these for training (preprocess.py will save the data files in pt format):
```
cd src/
python preprocess.py
python train.py
```

## Infer model 
You can modify the ```config/model_config.json``` file and run the following python script for infering the model with new input questions:
```
python run_model.py -config_path "config/model_config.json" -question "what was the result of the game with New York Jets?"
```
**Example**: with the current model_config.json and the above question in the command line the output will be:
```
SQL code: SELECT  `Result` FROM table WHERE `Opponent` = New York Jets
Execution result: w 20-13
```

## Use execution guidance for evaluating the model over test set
Evaluate the model over all questions from tables of the test set in WikiSQL:
```
cd src/
python evaluate.py -model_path ../pretrained/pretrain.pt -beam_search
```

