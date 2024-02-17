# deepfish

This is the deepfish repo to accompany the manuscript titled "Deep phenotypic profiling of neuroactive drugs in larval zebrafish". Manuscript URL: MANUSCRIPT_URL.

Environment setup: To train the Twin-NN and Twin-DN models, create a new conda environment from the provided requirements file as follows: 
conda create --name deepfish_env --file deepfish_env_req_simple.txt

Data: download data from provided zenodo repo: ZENODO_URL.

To train models: use provided config file or choose custom training parameters (config.yaml file). Activate conda environment: conda activate deepfish_env
Run main training loop: python TwinMain.py
