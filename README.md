# rl_autoaug

This is the repository for our ACM MM 2021 paper "Learning Sample-Specific Policies for Sequential ImageAugmentation". 
The code and dataset will be updated soon. 


### <a name="dependency"></a> Dependency
* Ubuntu ≥ 14.04
* Python ≥ 3.6.8
* tensorflow == 1.15.0

[comment]: <> (Configure the environment:)

[comment]: <> (> ```bash)

[comment]: <> (> pip install -r requirements.txt)

[comment]: <> (> ```)

### <a name="data"></a> Data
The dataset can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1XXelQAC-nXJBpYki7TwkO1jYLjBp1J0-?usp=sharing). 
We includes the CIRAR-10, CIFAR-100, Trash Dataset used in our experiments. Those datasets can be readily used in our code after extracting the *.zip files. 
The extracted folders also contains *.pkl file which is the index file we split the training and validation dataset. 

### <a name="trained_model"></a> Trained Model
The trained models for our experiments in the paper could be downloaded in [Google Drive](https://drive.google.com/drive/folders/1qeziowWu9YktZ9_CoD6pOLP_xVlJkWUQ?usp=sharing). The policy models are in folder "policy". 
And classifier is in folders "classifier".
 

[comment]: <> (### <a name="model_evaluation"></a> Model Evaluation)

[comment]: <> (1. To reproduce the evaluation traditional homography estimation methods in our paper:)

[comment]: <> (> ```bash)

[comment]: <> (> python test_homography_opencv.py --data_path [TEST_DATA_FOLDER] \)

[comment]: <> (>                                  --method [ESTIMATION_METHOD] \)

[comment]: <> (>                                  --ann_path [TEST_ANNOTATION_FOLDER] )

[comment]: <> (>                                  --scale 0.25)

[comment]: <> (> ```)

[comment]: <> (ESTIMATION_METHOD could be Identity or ORB2+RANSAC, which is correponded to our Table 1 in the paper.  )

[comment]: <> (2 To reproduce the evaluation of deep models in our paper:)

[comment]: <> (> ```bash)

[comment]: <> (> python test_homography_network.py --model_type [MODEL_TYPE] )

[comment]: <> (>                                   --model_file [PATH_TO_MODEL_WEIGHTS])

[comment]: <> (>                                   --data_path [TEST_DATA_FOLDER] )

[comment]: <> (>                                   --ann_path [TEST_ANNOTATION_FOLDER]  )

[comment]: <> (>                                   --scale 0.25)

[comment]: <> (> ```)

[comment]: <> (MODEL_TYPE should be CNN for model BASE, REG-P, REG-S, REG-T, and REG-ALL. MODEL_TYPE should be LSTM for model LSTM, LSTM-REG-ALL. )


### <a name="model_training"></a> Model training
The model training code will be released later. 
