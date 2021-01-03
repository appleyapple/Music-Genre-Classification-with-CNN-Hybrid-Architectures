# CRNN-for-Music-Genre-Classification

## Instructions to Generate Data:

  1. Have the /fma_small folder in the main folder of the repo
  
  2. Select the number of files to generate in data_setup.py
  
  3. Run python data_setup.py and all the folders and file will be generated

## To train the model:

  1. Specify parameters in load_data_generators.py
  
  2. Specify model in train.py - specify number of epochs, etc.
  
  3. Specify parameters in model.py - this is the file that contains all four models
  
  4. Run the command "python train.py"
  
 outputs your final model, best_model in .h5 format
 
## To test the model:
  
  1. Go to predict.py and specify which model to test, either model.h5, or best_model.h5
  
  2. Run "python predict.py"
  
 this outputs your final test accuracy and loss in the command line
 
*Make sure you have the .gitignore file so that the actual data does not get pushed onto the repo*


## Contributers:
  
  Andy Liu
  
  Henry Yip
