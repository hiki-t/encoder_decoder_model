
# install virtual env
python3 -m venv myenv

# save requirements.txt file to remote
pip install -r requirements.txt

# install jupyterlab
# pip install jupyterlab

# install Install IPython and ipykernel
pip install ipykernel

# Add Your Virtual Environment as a Jupyter Kernel
python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"

# to access target venv environment from vscode on remote server
# at first, need to setup with Python: Select Interpreter command from the Command Palette (Ctrl+Shift+P)
# then the target env is accessible through select Kernel
