# Bert-Edit-Unsup-TS
This repo contains the code for the paper "Improving Unsupervised Sentence Simplification Using Fine-Tuned Masked Language Models".

The original Edit-Unsup-TS code is shared [here](https://github.com/ddhruvkr/edit-unsup-ts), written in Python 3.7.6 and Pytorch 1.4.0.

## An overall look at our framework

![alt text](https://github.com/aminthemar/bert-edit-unsup-ts/blob/main/framework%20-%20diagram%20acl.png?raw=true)

## Data Selection for Fine-tuning

1. First, open the Google Colab notebook file "classification-with-roberta.ipynb" in the root folder or get a copy from here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1z3OF4dG2pNGxZQZbkpNlui-N6cmc9UEG?usp=sharing)

2. You will need two text files, one for every class (complex & simple) and each consisting of sentences in a line by line format.
<br/> Our training text files can be downloaded from [[here]](https://drive.google.com/drive/folders/11vx0iMOJxxXmAPrWQQ3MeolSRlRF137a?usp=sharing).

   > We cannot share the [Newsela](https://newsela.com/data/) training set since it must be requested from the owners.

3. After setting things up, you can either train a classifier on your own data or load one of our pre-trained models.
<br/> Our Newsela-trained and Wikilarge-trained RoBERTa simplicity classifiers can be found [[here]](https://drive.google.com/drive/folders/1w9pK5qOvN-PBCUOo4L5uGND5dbMl0kVc?usp=sharing).

4. Finally, you will need to input your fine-tuning text file (sentences in a line by line format) to get predictions from the classifier. 
<br/> Our prediction text files (from Wikilarge) can be found [[here]](https://drive.google.com/drive/folders/1SEDaXbh_sJX8XYfXOgdCcPkaLgzgVsze?usp=sharing).

   > Note: The prediction data does not need to have labels. However, our data is already classified only for extra analysis (optional).

## Fine-tuning BERT

Feed the resulting selections to the Google Colab notebook file "bert-fine-tune-atm.ipynb" to fine-tune a BERT masked language model.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LPFivtFx3se9U-k-VhSloSDzySVYMFRj?usp=sharing)

## Running Edit-Unsup-TS

Put the final BERT MLM folder in the "fine-tuned_models" folder of the Edit-Unsup-TS directory.
To start the simplification process, specify the directory of your BERT MLM in the "config.py" file. The rest of the process is identical to the [Edit-Unsup-TS guide](https://github.com/ddhruvkr/Edit-Unsup-TS/blob/master/README.md).

> The "data-simplification" folder is an important placeholder for the base Wikilarge or Newsela data. Details are explained in the original Edit-Unsup-TS guide.

## Evaluation

We evaluate [our simplification results](https://github.com/aminthemar/bert-edit-unsup-ts/tree/main/outputs) using the [EASSE framework](https://github.com/feralvam/easse).
