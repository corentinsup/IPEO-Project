################################
# GLACIER SEGMENTATION PROJECT #
################################

Please find our project through this GitHub URL: https://github.com/corentinsup/IPEO-Project

Dataset Location:   
    - The dataset generated for this project can be found in the ./dataset/clean/ directory for the training samples and in ./dataset/test for the testing.
    - All images and masks are directly hosted on the GitHub Repository, so you should be able to find them after cloning the Repository.

Environment and inference: 
    - The required inference notebook is inference.ipynb
    - Please click on the button at the top, "Open in Colab", which will redirect you to a functional notebook hosted on Google Colab.
    - You will also need to create a HuggingFace account if you want to run inference on our seconde model (based on DinoV3)
        - Here you will find instruction for generating an access token: https://huggingface.co/docs/hub/security-tokens
        - You will also Need to request the permission to Meta to be able to use their models, which is explained in the notebook.
    - Simply follow the instruction of the notebook, which will install the requirements defined in requirements.txt (this might take up to 10minutes since it builds libraries)
    - The HuggingFace step is not necessary if you want to use uniquely the U-Net++ model


Data provenance:
    - The satelite imagery was extracted from Sentinel-2, using the Google Eart Engine API.