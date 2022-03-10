# ELECTRA MODEL CHANGES FOR CPU FALLBACK

## REQUIREMENTS- (INSTALL MODULES)
```bash
pip install tokenizers==0.7.0
pip install boto3
pip install filelock
```
## Changes made to the `/cpu-fallback/ELECTRA/run_pretraining.py` file- The main training script.
```
* Modified the code to remove gpu 
* Added a parameter use_hpu to handle run on hpu
```

## Changes made to the `/cpu-fallback/electras.sh` file- The wrapper shell script.
```
* Added support to handle flag "hpu" passed to the script to run on HPU, else if "cpu" or no parameter is passed to the script, then it run on CPU
* Changed it to run using mpirun instead of hororvodrun is num_hpu value is provided. Default set to 0.
* The values of different hyper-parameters are changed to low values as compared to the original values to handle the execution on CPU and resolve memory issues.
```

## Run training
(1) Run on hpu
```
./electra.sh hpu
``` 

(2) Run on cpu
```
./electra.sh cpu
```
