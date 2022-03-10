# Demonstration of Running Multiple Trainings in Parallel

This directory provides some scripts to demonstrate how to run multiple training jobs in parallel. The training jobs mentioned here utilize multiple Gaudis but not all of the Gaudis in the same server. The flow and configuration to run with partial Gaudis is basically the same as running with all Gaudis on the machine. except few additional configurations to avoid contention of the same resource.

## multitask_resnet.sh

This script invokes 2 resnet50 jobs in parallel, and each uses 4 Gaudis. Below are some environment variables and python script arguments need to explicitly specified as different value for the 2 jobs:

### HLS1_MODULE_ID_LIST
environment variable for the list of module IDs, composed by a sequence of single digit integers. There same integer shouldn't be used by multiple jobs running in parallel
For jobs with 4 Gaudis, we suggest set this to "0123" or "4567"
For jobs with 2 Gaidis, we suggest set this to "01", "23", "45", or "67"

### -md/--model_dir
python script argument for the model directory, needs to be specified to different paths for different jobs