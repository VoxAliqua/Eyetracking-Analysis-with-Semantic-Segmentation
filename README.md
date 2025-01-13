
# Automated annotation of experimental material and analysis of an eyetracking experiment

### Semantic Segmentation models
Semantic Segmentation package, trained transformers and project cloned from MIT CSAIL lab - https://github.com/CSAILVision/semantic-segmentation-pytorch \
Pretrained models were found at: http://sceneparsing.csail.mit.edu/model/pytorch

### Eyetracking Pipeline
#### Semantic Segmentation:
* Splitting video material into individual frames
* Batch segmentation of the individual frames using a pretrained model; each pixel receives a semantic label 

#### Eyetracking Analysis:
* initial conversion of raw eyelink files to panda dataframes. 
* Automated recognition of starting points of experiments, and associated viewing material, ie. the movie clips.
* Matching timestamps of eyetracking to video frames
* Matching corresponding pixel and associated segmentation label
* Assigning experimental category based on segmentation label. 

#### Notes
Please be advised that this code was created under considerable time pressure and is not done in a professional manner, nor did I ever really intend to make this code accessible to others outside of my research project. What we have here are the bones, and a bunch of basics are missing like a requirements.txt, a hardware description, and how to set up CUDA etc\.

Should I ever have the time and the motivation I will set this up properly.
