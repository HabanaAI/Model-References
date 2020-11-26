# Stereo repo

## Welcome Habana!

To run our training example you should do the following:
1. Copy repository from s3 to local storage
`aws s3 cp s3://mobileye-habana/mobileye-team-stereo/stereo-habana.tar.gz <local path>` 
2. `cd <local path>`
`tar -xzf stereo-habana.tar.gz`
`cd stereo-habana`
3. Create venv `./create_venv_habana.sh <python 3.6 interpeter> <venv base> <venv name>`
4. Activate it `source <venv base>/<venv name>/bin/activate.csh`
5. Run train example notebook (jupytext style so you can also run it as a script)
`python train_example.py`

# Notes
1. Last cell of the notebook set the training parameters like `model_dir`, `batch_sz` etc.
2. Note multi-gpu option, we use it for multi-gpu machines
