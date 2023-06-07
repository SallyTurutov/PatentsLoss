# PatentsLoss
Generating Optimized Molecules without Patent Infringement

# Installation and Training:
Each model is handled based on its own instruction as stated in the original model.
The data used for the patents loss can be found in the data folder of each model.
The code with the method is property_handler.py - the function get_similar_patents_list() is responsible to get the patents used in the adjusted loss. The method to train the model is modified by choosing the wanted method in this function. Every model holds its own property_handler.py file.

# The overall patents dataset:
The overall patents dataset can be found in http://ftp.ebi.ac.uk/pub/databases/chembl/SureChEMBLccs/. In order to handle the overall dataset:
1. Install conda / minicaonda
2. From the UGMMT folder run:\
    i. conda env create -f environment.yml\
    ii. conda activate UGMMT
3. Download SureChEMBL dataset from: http://ftp.ebi.ac.uk/pub/databases/chembl/SureChEMBLccs/ \
    i. Lockate the downloaded dataset in dataset/SureChEMBL \
    ii. From the main folder run: python handle_patents_dataset.py \
    iii. Move the generated SureChEMBL.txt file to dataset/QED
 
 However, notice that all the dataset used for training is already found in data/PL/ folder of each model.
