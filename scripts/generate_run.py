import yaml # for parsing the yaml files
import os # for running shell commands
import copy # for copying objects
import glob # for globbing the files in the directory
from itertools import product


def generate_bash_script(experiment_name, output_bash_dir, config, lambda_value, lambda2_value, conda_env, partition, gpu_type, batch_size, wall_time='24:00:00'):
    bash_dir   =f'{output_bash_dir}/{experiment_name}.sh'

    bash_script = f'''#!/bin/bash -l
#SBATCH --job-name={experiment_name}
#SBATCH --nodes=1

# For {partition}
#SBATCH --partition={partition}
#SBATCH --gres={gpu_type}
#SBATCH --time={wall_time}
 
#  SBATCH --output=path/to/workspace/scripts/train_scripts/logs/{experiment_name}.out
#  SBATCH --error=path/to/workspace/scripts/train_scripts/logs/{experiment_name}.err


# change to the working directory
main_hpc_dir="${{main_hpc_dir:-path/to/workspace}}"
repo_dir=$main_hpc_dir"/repos/TokenCompression"
cd $repo_dir

# activate the conda environment
source ~/miniconda3/bin/activate {conda_env}"

experiment_name="{experiment_name}"

data_dir=$main_hpc_dir"/data/images"
data_csv=$repo_dir"/datasets/csv/train_256"

echo "Running experiment"
config_name="configs/models/{config}"
config_file=$repo_dir"/"$config_name
batch_size={batch_size} 
epochs=100

python scripts/train.py --dataset $data_dir --config $config_file --dataset_csv $data_csv --epochs $epochs --exp $experiment_name --learning_rate 1e-4 --clip_max_norm 10  --batch_size $batch_size --test_batch_size $batch_size


'''

    os.makedirs(os.path.dirname(bash_dir), exist_ok=True)    
    with open(bash_dir, 'w') as f:
        f.write(bash_script)


if __name__ == '__main__':
    main_path = os.path.dirname(os.path.abspath(__file__))
    output_bash_dir = f'{main_path}/train_scripts'
    os.makedirs(output_bash_dir, exist_ok=True)

    # experiments_names = os.listdir(f'{os.path.dirname(main_path)}/configs')
    # experiments_names = [x for x in experiments_names if x.endswith('.yaml')]
    # experiments_names_to_remove = ['default.yaml',]
    # experiments_names = [x for x in experiments_names if x not in experiments_names_to_remove]
    # experiments_names = [x.split('.')[0] for x in experiments_names]
    experiments_names = ['opod_tic_small_vq_128_vae_16']
    mse_values = [0.01] 
    loss = ['mse']

    # for H100
    for experiment_name in experiments_names:
        conda_env  = "TokenCompression"
        gpu_type   = "gpu:a100:2 -C a100_80"
        # gpu_type   = "gpu:h100:1"
        batch_size = 40
    
        partition= gpu_type.split(':')[1]
        output_bash_dir_loss = f'{output_bash_dir}/{experiment_name}'
        os.makedirs(output_bash_dir_loss, exist_ok=True)
        wall_time = '72:00:00' if 'h100' in gpu_type else '24:00:00'
        for mse in mse_values:
            generate_bash_script(f'{experiment_name}_{mse}', output_bash_dir_loss, experiment_name, mse, mse, conda_env, partition, gpu_type, batch_size, wall_time)

print('Done')