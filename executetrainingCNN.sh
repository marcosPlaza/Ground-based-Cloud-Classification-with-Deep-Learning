#!/bin/bash
python TrainingPipelineCNN.py True True False True "./Models/SwimCat_lre-5_500.h5" "/Users/marcosplazagonzalez/Desktop/Ground-based_CloudClassification/Datasets/Splitted" "./Models/FabraTest5Clases.h5" "./Historics/FabraTest5Clases_History.pkl" 600 8 1e-5 "./Data/swimcatdataset.data" False True False

#   tensorboard = sys.argv[1] == 'True' 
#   save_historic = sys.argv[2] == 'True'
#   scheduler = sys.argv[3] == 'True'
#   pretrained = sys.argv[4] == 'True'
#   pretrained_model = sys.argv[5]
#   abs_path = sys.argv[6]
#   model_path = sys.argv[7]
#   history_path = sys.argv[8]
#   epochs = int(sys.argv[9])
#   batch_size = int(sys.argv[10])
#   initial_lr = float(sys.argv[11])
#   save_path = sys.argv[12]
#   data_from_file = sys.argv[13] == 'True'
#   data_augmentation = sys.argv[14] == 'True'
#   early_stopping = sys.argv[15] == 'True'