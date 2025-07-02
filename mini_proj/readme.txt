
# make sure you are in right directory
# which is where all essential data exists

# install virtual env
python3 -m venv myenv

# save requirements.txt file to remote
pip install -r requirements.txt

# install jupyterlab IPython and ipykernel
pip install jupyterlab ipykernel

# Add Your Virtual Environment as a Jupyter Kernel
python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"

# to access target venv environment from vscode on remote server
# at first, need to setup with Python: Select Interpreter command from the Command Palette (Ctrl+Shift+P)
# then the target env is accessible through select Kernel

# login to huggingface repo with hf token
huggingface-cli login
# login to wandb account with api
wandb login
