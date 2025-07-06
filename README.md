-----------------------------------------------------------------------

This project provides a single ready to run script that:
	- Loads a pretrained GeoLDM model
	- Generates 3D molecular conformers for the e-Drug3D dataset
	- Converts to RDKit molecules (with automatic bond assignment)
	- Computes and saves evaluation metrics and plots

---------------------------- Installation ------------------------------

# Clone the GeoLDM repo
We already cloned the GeoLDM repository and made the changes necessary for our adaptation and use

# Install core dependencies
pip install
	- numpy
	- scipy
	- matplotlib
	- seaborn 
	- torch
	- tqdm
	- rdkit-pypi


---------------------------- Configuration ----------------------------

In final_script.py, set the correct paths for your environment:

# Path to pretrained model arguments (args.pickle)
args_path = '/full/path/to/drugs_latent2/args.pickle'

# Path to pretrained model weights (generative_model.npy)
model_path = '/full_path_to/drugs_latent2/generative_model.npy'

# Path to your real 3D conformers NumPy file
data_file  = '/full_path_to/e-drugs_conf.npy'

All other parameters (sample sizes, batch sizes, seed) work out of the box.

---------------------------- Run ----------------------------
python final_script.py
