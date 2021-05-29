# ADFAR

This is the code for our long paper:

Defending Pre-trained Language Modelsfrom Adversarial Word Substitution Without Performance Sacrifice

Accepted to Findings of ACL 2021.

## Dependencies
[TextFooler](https://github.com/jind11/TextFooler)

## Usage

Download [counter-fitted-vectors.txt](https://drive.google.com/file/d/1bayGomljWb6HeYDMTDKXrh0HackKtSlx/view) and then put it in '/src/TextFooler/'. 
Download [USE](https://tfhub.dev/google/universal-sentence-encoder-large/3) and then put it in '/src/TextFooler/'.

Then you can run the run_MR_pipeline.py to see how the pipeline works and gets the defense performance on MR dataset as an example.
