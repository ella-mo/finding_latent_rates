#!/bin/bash
#SBATCH -p batch
#SBATCH -t 0:30:00
#SBATCH --mem=70G
#SBATCH -J run_data_inspection_main
#SBATCH -o logs/run_main_%j.out
#SBATCH -e logs/run_main_%j.err

if [ ! -d "logs" ]; then
  mkdir logs
fi

module load miniforge3/25.3.0-3
source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh
module load cuda/11.8
eval "$(conda shell.bash hook)"
conda activate csci1470

well_list=('C2' 'C3' 'D1' 'D4' 'D5' 'D6' 'D7' 'D8')
for well in "${well_list[@]}"; do
  echo "=========================================="
  echo "Running data inspection for well $well"
  echo "=========================================="
  python main.py '/oscar/home/emohanra/scratch/lizarraga/finding_latent_rates/mea-mua-analysis/files' '/oscar/home/emohanra/scratch/lizarraga/waveformVariability/bin_files' -d 83 -r 6 9 -w $well -vt -n 16
done