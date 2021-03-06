# Latent Timbre Synthesis - Pytorch version

Pytorch implementation of Latent Timbre Synthesis. Original repo is available at: https://github.com/ktatar/latent-timbre-synthesis

Latent Timbre Synthesis (LTS) is a deep learning audio synthesis architecture that learns from audio sample dataset.

Project page: https://kivanctatar.com/latent-timbre-synthesis

## Latent Timbre Synthesis using Variational Auto-Encoders
We combine Variational Auto-Encoders with inverse audio synthesis to generate a latent audio space for musical applications.

## Installation 

The training and the synthesis engines are working on Python. Currently, we have an app called interpolate_two which uses Max 8 for the user interface and interaction with the engine. 

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


For using the interpolate_two app

- Install python-osc for osc communication

`pip install python-osc`

- Install sounddevice for Audio IO in python

`conda install -c conda-forge python-sounddevice`
    
  - If you get "Module not found" error after the  sounddevice installation:
    - First try to close and open your terminal, and then run it again.
      - If you still have the error, try this fix:
            
        `pip uninstall sounddevice`
            
        `pip install sounddevice`

4- Max 8 Libraries

- For the app interpolate_two, install AudioMix from Max 8 Package Manager. 

Enjoy...

## Using interpolate_two app

This application provides a framework to synthesize sounds using interpolation and extrapolation with two sounds. The framework requires a trained model to generate sounds using the latent audio space. The user can select the duration of the generated sound, and can choose an excerpt from two audio files with the same duration. The algorithm uses these two excerpts for synthesis with interpolation and extrapolation in the timbre space. The interpolation amount sets how much of the latent vector is copied from one of the audio excerpts. For example, 30% interpolation is adding 30% of the latent vectors of the first audio and 70% of the latent vectors of the second audio. The percentages above 100 or below 0 corresponds to extrapolations. For instance, 120% is moving 20% away from the second audio in the direction the latent vector that links audio 1 and audio 2. 

The algorithm synthesizes the audio by calculating every audio frame using inverse synthesis from a generated spectrogram. Hence, the user sets interpolation amounts for each vector. With the interface, the user can draw an interpolation curve to change the interpolation percentage in time.

### How to run

1- Open your terminal, and activate lts environment,

`conda activate lts`

2- Change your working directory to the interpolate_two app

In the code below, change the [my-latent-timbre-synthesis-master] section to the path of the latent-timbre-synthesis-master folder on your computer. 
You can find that path by,

- On Windows: 
    - hold shift and right-click on the interpolate_two folder in the repo, and then select "copy as path"
    - `cd [paste-the-path-here]`
    - For example, it would look like this on my PC: `cd C:\Users\tatar\latent-timbre-synthesis\app\interpolate_two`
- On Mac: 
    - right-click on the folder, hold alt (option) key and select "Copy ... as pathname"
    - `cd [paste-the-path-here]`
    - - For example, it would look like this on my Mac: `cd /Users/tatar/Desktop/latent-timbre-synthesis\app\interpolate_two`
- On Linux: if you can use linux, you got this (:

3- Run main.py in the interpolate two folder:

`python main.py`

4- Open Max patch max\_ui.maxpat

### The framework

interpolate\_two has two engines, Max GUI and python engine. Python engine runs the deep learning model and handles audio feature extraction as well as inverse synthesis that generates audio from magnitude spectrogram using phase estimation. The python engine reacts to OSC messages coming from the max\_ui.maxpat 

### The GUI in MAX 8

![alt text](https://gitlab.com/ktatar/latent-timbre-synthesis/-/raw/stable/images/interpolate_two_ui.jpg "interpolate_two GUI")

A) Titles & Credits 

B) Drop your dataset folder here 

C) Drop your run folder here to load a trained model

D) and E) Select two audio files from your dataset to choose excerpts. The file dropdown menu allows to select an audio file from the dataset. "Zoom to selection" sets the view to the selection area. Clicking crtl (or cmd on macos) and then dragging the mouse up&down on the waveform views applies zoom in & out.  

F) The curve to apply interpolation and extrapolation, where x is the time and y sets the interpolation percentage for a frame. When both waveform views are zoomed to the selection, the x-axis of interpolation curve corresponds to the x-axis of the waveform. The interpolation curve view is a waveform~ object. The Vertical Zoom parameter in the inspector sets the maximum interpolation amount. The default maximum is 1.3; hence, [1.0,1.3] and [-1.0, -1.3] is the extrapolation regions. It is possible to extrapolate even more by changing the vertical zoom parameter; however, the higher amounts are likely to give audible distortions. 

H) This section send messages to the python engine to handle output generation. "Generate & Play" initiates the python engine to synthesize a sound using the current interpolation curve and the audio selections. "Play Again" plays the previous generated sound, without going through the deep learning calculation. "STOP" immediately stops the audio coming out of the python engine. Normalize on/off button sets the audio output volume normalization on and off. when the normalization is set to on, the maximum of the generated audio is set 0.0 db. Phase iterations sets the number of iterations of the Fast Griffin-Lim algorithm. Higher number of iterations (max. 64) gives better results; however, the calculation takes significantly longer. The phase estimation algorithm takes the bottleneck of computational complexity of this framework. Still, the calculation of the audio takes 50% of the audio duration with phase iteration set to 1. That is, calculating a 2-second sound takes around 1-second on a latest, high-end computer. 

I) This section handles the audio output specs of the audio engine of python server. The available audio devices are automatically send from python engine while opening. User can also click "Get Available Audio Devices for Python Engine" if a new audio device is plugged after initiating the python engine. Input and Output dropdown menus sets the audio i&o of the python engine. Lastly, "set previous files again" is for occasions where the python engine crashes. This button sets the audio excerpts in the python engine. 

### If your python engine crashes while working...

You can easily set everything from where you left by clicking,

1- The dataset message box under "Drop Dataset Folder"

2- The run folder message box under "Drop Run Folder"

3- "set previous audio files again" message box in the Python Audio section

# Training your own models 

Let's create a dataset to before the VAE training:

- Gather some audio files
- Create a folder with any name. This is your dataset folder.
- In your dataset folder, create a folder called audio. Move all your audio files in that folder. 

We need to create the CQT vectors saved in .npy numpy files. To do so, we will use the create_dataset.py script. 
This script expects a config file where audio parameters are given. Let's say you would like to use [lts_repo]/default.ini as your config file

` python [lts-repo]/utils/create_dataset.py --config [lts_repo]/functional/timbre_vae/default.ini`

There are three different scripts that you can use to train your own models:

1- functional/timbre_vae/train.py

2- timbre_vae2/timbre_vae/train.py

3- timbre_cvae/timbre_vae/train.py

The models generated by any of these scripts are compatible with the interpolate_two application. 
We recommend the first script, as it is the most simple architecture that gives the best results. 

train.py scripts expects a config file. In each folder, there is an example config file with the name default.ini.
Change the dataset parameter to the path of your dataset folder. Leave the workspace parameter empty. 
The description parameter creates a folder where the models generated during the runs will be saved. 
The configs and the description parameter is especially usefull if you run several trainings in different machines concurrently 

Hence, you can use it as 

`python [lts_repo]/functional/timbre_vae/train.py --config ./default.ini`

