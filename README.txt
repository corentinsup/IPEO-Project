################################
# GLACIER SEGMENTATION PROJECT #
################################

Please find our project through this GitHub URL: https://github.com/corentinsup/IPEO-Project

Dataset Location:   
    - The dataset generated for this project can be found in the ./dataset/clean/ directory for the training samples and in ./dataset/test for the testing.
    - All images and masks are directly hosted on the GitHub Repository, so you should be able to find them after cloning the Repository.

Environment and inference: 
    - The required inference notebook is inference.ipynb.
    - Please click on the button at the top, "Open in Colab", which will redirect you to a functional notebook hosted on Google Colab.
    - You will also need to create a HuggingFace account if you want to run inference on our seconde model (based on DinoV3).
        - Here you will find instruction for generating an access token: https://huggingface.co/docs/hub/security-tokens.
        - You will also need to request the permission to Meta to be able to use their models, which is explained in the notebook.
    - Simply follow the instruction of the notebook, which will install the requirements defined in requirements.txt (this might take up to 10minutes since it builds libraries).
    - The HuggingFace step is not necessary if you want to use uniquely the U-Net++ model.

Other files/code:
    - The dataset generation code is in the "Dataset pipeline.ipynb" notebook. Note that you will need a working python environment linked to your Google Earth engine account.
    - "cross_val.py" is the script used to run our K-Fold Cross-validation experiments to get the hyper-parameters.
    - "k-fold_epochs.py" is the K-Fold Cross-validation used to fine the appropriate number of epochs to run our model.
    - "train.py" is the final script that we used to obtain the weights with the approriate hyper-parameters and number of epochs.
    - "./scitas/" contains a sample script to run a training on scitas.
    - "./glamos/" contains the Swiss Glacier Inventory dataset of Glamos (used by the dataset generation notebook).

Data provenance (more detailed in report):
    - The satelite imagery was extracted from Sentinel-2, using the Google Eart Engine API.
