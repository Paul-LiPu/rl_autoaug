# rl_autoaug

This is the repository for our ACM MM 2021 paper "Learning Sample-Specific Policies for Sequential ImageAugmentation".


### <a name="dependency"></a> Dependency
* Ubuntu ≥ 14.04
* Python ≥ 3.6.8
* tensorflow == 1.15.0

[comment]: <> (Configure the environment:)

[comment]: <> (> ```bash)

[comment]: <> (> pip install -r requirements.txt)

[comment]: <> (> ```)


### <a name="clone"></a> Clone this repository
> ```bash
> git clone https://github.com/Paul-LiPu/rl_autoaug.git
> ```

### <a name="data"></a> Data
The dataset can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1XXelQAC-nXJBpYki7TwkO1jYLjBp1J0-?usp=sharing). 
We includes the CIRAR-10, CIFAR-100, Trash Dataset used in our experiments. Those datasets can be readily used in our code after extracting the *.zip files. 
The extracted folders also contains *.pkl file which is the index file we split the training and validation dataset. 

Download those files and extract them in 'CIFAR100_wrn28-10/data'
> ```bash
> cd CIFAR100_wrn28-10/data
> unzip cifar100.zip
> ```

### <a name="trained_model"></a> Trained Model
The trained models for our experiments in the paper could be downloaded in [Google Drive](https://drive.google.com/drive/folders/1qeziowWu9YktZ9_CoD6pOLP_xVlJkWUQ?usp=sharing). The policy models are in folder "policy". 
And classifier is in folders "classifier".


### <a name="model_training"></a> Model training
The iterative training of classifier and policy network can be done by
> ```bash
> cd CIFAR100_wrn28-10
> python workflow.py
> ```