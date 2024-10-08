#!/usr/bin/env bash
#SBATCH -A NAISS2024-22-16 -p alvis
#SBATCH -t 3-00:00:00
#SBATCH --gpus-per-node=A100:1
#SBATCH --mail-user=tatar@chalmers.se
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

module load virtualenv/20.23.1-GCCcore-12.3.0 PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 SciPy-bundle/2023.07-gfbf-2023a matplotlib/3.7.2-gfbf-2023a torchvision/0.16.0-foss-2023a-CUDA-12.1.1  
sleep 5
source /cephyr/users/tatar/Alvis/ltsp/bin/activate

cd /cephyr/users/tatar/Alvis/code/latent-timbre-synthesis-pytorch

# python create_dataset.py --config ./default-alvis.ini

python ./train.py --config ./default-alvis-2.ini
