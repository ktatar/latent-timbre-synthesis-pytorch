# Latent Timbre Synthesis - Pytorch version

Pytorch implementation of Latent Timbre Synthesis. Original repo is available at: https://github.com/ktatar/latent-timbre-synthesis

Latent Timbre Synthesis (LTS) is a deep learning audio synthesis architecture that learns from audio sample dataset.

Project page: https://kivanctatar.com/latent-timbre-synthesis

## Latent Timbre Synthesis using Variational Auto-Encoders
We combine Variational Auto-Encoders with inverse audio synthesis to generate a latent audio space for musical applications.

## Installation 

### Python 

1- Conda 
Download and install Anaconda for Python 3.7 for your os:
https://www.anaconda.com/distribution/

- Create a new python 3.7 environment
Please open your terminal (Anaconda Command Prompt for Windows),

`conda create -n lts python=3.7`

- Activate your conda environment before continuing the following steps:

`conda activate lts`
    
If you have an issue with the command above on macos, try: 
`source activate lts`

2- Pytorch
Install pytorch using conda: https://pytorch.org/. If you have an NVIDIA GPU, please follow the GPU guide. For all other hardware, please follow the CPU option. The code is compatible with both GPU and CPU. The inference can run without GPU; however, the training would take too long on CPU. I strongly recommend using a GPU if you plan to train your own models.

3- Python Libraries

- Install librosa for audio feature extraction

`pip install librosa`

train.py scripts expects a config file. In the main folder, there is an example config file with the name default.ini.
Change the dataset parameter to the path of your dataset folder. Leave the workspace parameter empty. 
The description parameter creates a folder where the models generated during the runs will be saved. 
The configs and the description parameter is especially usefull if you run several trainings in different machines concurrently 

Hence, you can use it as 

`python [lts_repo]/functional/timbre_vae/train.py --config ./default.ini`

