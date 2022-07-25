# Demonstration of Running Multiple Trainings in Parallel

This directory provides some scripts to demonstrate how to run multiple training jobs in parallel. The training jobs mentioned here utilize multiple Gaudis but not all of the Gaudis in the same server. The flow and configuration to run with partial Gaudis is basically the same as running with all Gaudis on the machine, except from few additional configurations to avoid contention of the same resource.

## multitask_resnet.sh

This script invokes 2 resnet50 jobs in parallel, and each uses 4 Gaudis. Below are some environment variables and python script arguments that need to be explicitly specified as different values for both jobs.

### HABANA_VISIBLE_MODULES
An environment variable for the list of module IDs, composed by a sequence of single digit integers. The same integer should not be used by multiple jobs running in parallel: 
For jobs with 4 Gaudis, it is recommended to set this to "0,1,2,3" or "4,5,6,7"
For jobs with 2 Gaudis, it is recommended to set this to "0,1", "2,3", "4,5", or "6,7"

### -md/--model_dir
Python script argument for the model directory needs to be specified to different paths for the different jobs. 