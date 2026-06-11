## Examples

### Environment
Create and activate the venv specified in environment.yml.

### Train a new U-Net
To train a new U-Net, use train_new_unet.py and update the paths for training and validation data and images. See example of the expected input formats for csvs and NIfTI images in datasets/simulated_example/.

Training was run on an HPC cluster using Slurm. An example job submission script is provided in start_unet.sh. To submit a job:
´´´
sbatch start_unet.sh
´´´
Adjust the resource parameters (e.g. --gres=gpu, --time) in the script to match your cluster's configuration.

The training process can be monitored using tensorboard.
´´´
tensorboard --logdir outputs/logs/[NAME_OF_RUN] --port 8008
´´´
<p align="center">
<img width="900" alt="git_tensorboard" src="https://github.com/user-attachments/assets/987f7cfd-773a-41b0-aeb7-c0427df0fdc4" />
</p>

### Evaluate a U-Net
To evaluate a trained U-Net, use evaluate_unet.py and update that paths for training and test data and images. The training data is only used for imputation and scaling of test data. Also specify the weights for the model saved from training.

The output is 
1) a csv with summary image metrics and suvr in Braak regions comparing true and synthetic images.
2) synthetic tau-PET for the full test set saved as NIfTI files.

