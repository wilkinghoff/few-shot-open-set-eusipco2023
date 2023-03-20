# few-shot-open-set-eusipco2023

## Description

Few-shot open-set classification system for https://zenodo.org/record/3689288#.Y5nGzn2ZNaQ. The system is an adaption of the anomalous sound detection system provided here: https://github.com/wilkinghoff/icassp2023/.

## Instructions

The implementation is based on Tensorflow 2.3 (more recent versions can run into problems with the current implementation). Just start the main script for training and evaluation. To run the code, you need to download the meta data, pattern sounds and unwanted sounds and store the directories in the same directory as the code contained in this repository.

## Reference

When reusing (parts of) the code, a reference to the following paper would be appreciated:

@unpublished{wilkinghoff2023using,
  author = {Wilkinghoff, Kevin and Fritz, Fabian},
  title  = {On Using Pre-Trained Embeddings for Detecting Anomalous Sounds with Limited Training Data},
  note   = {Submitted to 31st European Signal Processing Conference (EUSIPCO)},
  year   = {2023}
}
