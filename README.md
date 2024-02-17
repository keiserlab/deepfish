# DeepFish

## This is the deepfish repo to accompany the manuscript titled "Deep phenotypic profiling of neuroactive drugs in larval zebrafish". Manuscript URL: MANUSCRIPT_URL.

**Environment setup**: To train the Twin-NN and Twin-DN models, create a new conda environment from the provided requirements file as follows: 
conda create --name deepfish_env --file deepfish_env_req_full.txt

**Data**: download data from provided zenodo repo: ZENODO_URL. Put the data in the 'Data/' directory or whichever directory you prefer (just make sure to edit the path in the config file) 

**To train models**: use provided config file or choose custom training parameters (config.yaml file). Activate conda environment: conda activate deepfish_env. We provide the pre-enumerated training and test pairs in the data repo as numpy arrays as described in the methods section. You can use a different train/ test splitting approach, just save the resulting pairs to a numpy array and place in the Data directory.

Run main training loop: python TwinMain.py
