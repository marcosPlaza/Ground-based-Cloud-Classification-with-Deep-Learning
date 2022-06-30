#!/bin/bash
python TrainingPipelineCNN.py True True False False "./Models/Test10_CCSN_TrainTest_Clean_1e-5_noPre_noAug.h5" "/Users/marcosplazagonzalez/Desktop/Ground-based_CloudClassification/Datasets/FabraClouds256_TrainTest_Clean/train" "./Models/Test23_FabraClouds256_TrainTest_Clean_1e-3_PreImagenet_noAug.h5" "./History/Test23_FabraClouds256_TrainTest_Clean_1e-3_PreImagenet_noAug_History.pkl" 200 8 1e-3 "./Data/ccsn256clean_train.data" False False True True

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
#   imagenet = sys.argv[16] == 'True'