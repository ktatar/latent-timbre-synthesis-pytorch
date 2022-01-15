#!/bin/bash
#SBATCH --gres=gpu:v100l:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=72000M        # memory per node
#SBATCH --time=10-22:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --mail-user=ktatar@sfu.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
 
module load nixpkgs/16.09  gcc/7.3.0  python/3.7.4 cuda/10.0.130 cudnn/7.6
source /home/$USER/pt17/bin/activate
sleep 5
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$EBROOTCUDA/lib64:$EBROOTCUDNN/lib64

# python /home/ktatar/scratch/cqtgan/create_dataset.py --config /home/ktatar/scratch/cqtgan/default-cedar.ini
# wait
python /home/ktatar/scratch/latent-timbre-synthesis-pytorch/train.py --config /home/ktatar/scratch/latent-timbre-synthesis-pytorch/default-cedar.ini
