#!/bin/bash
#SBATCH --job-name=mask_pretrained_job                        # Job name
#SBATCH --output=maskgit_output_pret.txt                       # Standard output
#SBATCH --error=maskgit_error_pret.txt                         # Standard error
#SBATCH --time=24:00:00                           # Time limit (hh:mm:ss)cv_env/my_job.sbatch
#SBATCH --partition=lmbhiwi_gpu-1080ti # Partition name
#SBATCH --ntasks=1                                # Number of tasks
#SBATCH --cpus-per-task=1                         # Number of CPUs per task
#SBATCH --mem=32G                                  # Memory per node
#SBATCH --gres=gpu:2

nvidia-smi

export PYTHONPATH=/home/mutakeks:$PYTHONPATH

export PYTHONPATH=$PYTHONPATH:/home/mutakeks/final_titok/MastersProject_TiTok

# srun python main_pl.py --base configs/maskgit.yaml -t True --n_gpus=2 --logdir /work/dlclarge2/mutakeks-titok/maskgit_1d_dino64_150

srun python -m torch.distributed.launch --nproc_per_node=2 main_pl.py \
    --base /work/dlclarge2/mutakeks-titok/maskgit_1d_dino64_150/config.yaml \
    -t True --n_gpus 2 --resumes /work/dlclarge2/mutakeks-titok/maskgit_1d_dino64_150/checkpoints/checkpoints/last.ckpt 






