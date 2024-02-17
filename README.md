# DeepFish

## This is the deepfish repo to accompany the manuscript titled "Deep phenotypic profiling of neuroactive drugs in larval zebrafish". Manuscript URL: [MANUSCRIPT_URL].

### **Environment setup:** 
&nbsp;&nbsp;&nbsp;&nbsp; To train the Twin-NN and Twin-DN models, create a new conda environment from the provided requirements file as follows: `conda create --name deepfish_env --file deepfish_env_req_full.txt`.\
The code was tested and model training was performed on NVIDIA a GeForce GTX 1080 Ti GPU and a CentOS Linux kernel 3.10.0 operating system with an x86-64 architecture.

### **Data:**
&nbsp;&nbsp;&nbsp;&nbsp; Download data from provided zenodo repo: [ZENODO_URL]. Put the data in the 'Data/' directory or whichever directory you prefer (just make sure to edit the path in the config file) 

### **To train models:**
&nbsp;&nbsp;&nbsp;&nbsp; Use provided config file or choose custom training parameters (config.yaml file). Activate conda environment: `conda activate deepfish_env` \
We provide the pre-enumerated training and test pairs in the data repo as numpy arrays as described in the methods section. You can use a different train/ test splitting approach, just save the resulting pairs to a numpy array and place in the Data directory.\
Run main training loop: `python TwinMain.py`
