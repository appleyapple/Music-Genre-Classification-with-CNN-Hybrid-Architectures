# CRNN-for-Music-Genre-Classification

Instructions to Generate Data:
  Have the /fma_small folder in the main folder of the repo
  Select the number of files to generate in data_setup.py
  run python data_setup.py and all the folders and file will be generated

To train the model:
  specify parameters in load_data_generators.py
  specify model in train.py - specify number of epochs, etc.
  specify parameters in model.py - this is the file that contains all four models
  run the command "python train.py"
  
 outputs your final model, best_model in .h5 format
 
To test the model go to predict.py and specify which model to test, either model.h5, or best_model.h5
  run "python predict.py"
  
 this outputs your final test accuracy and loss in the command line
 
*Make sure you have the .gitignore file so that the actual data does not get pushed onto the repo*


Contributers:
  Andy Liu
  Henry Yip
