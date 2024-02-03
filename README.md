# Planty
This is an implementation of a plant `image classifier` which classifies 102 categories of plants
![Plants](assets/Flowers.png)

### Architecture
___
This implementation supports the `resnet50` and `vgg16` architectures and has a output layer with 102 units representing each of the possible plant categories.

### Implementation
___
The `image classifier` was built using Pytorch and was tested on a dataset which you can by running `python flowers.py` in the command line. 

### Use
___
To begin training a model run `python train.py flowers` where `flowers` is the dataset containing 102 categories of plants and to make a prediction run `python train.py filepath model_checkpoint` where `filepath` is the path to the image you want to make a prediction on and `model_checkpoint` is your saved model.

The program also supports several command-line arguments:

 **Training**
`python train.py data_dir --arch [ARCH]` - specifies architecture to train model on by default it is set to vgg16.
`python train.py data_dir --save_dir [SAVEDIR]` - saves trained model in specified directory.
`python train.py data_dir --learning_rate [LR]` - specifies training learning rate by default it is set to 0.001.
`python train.py data_dir --epochs [EPOCHS]` - specifies number of training epochs by default it is set to 20.
`python train.py data_dir --hidden_units [HIDDEN]` - specifies number of hidden units for training.
`python train.py data_dir --gpu` - utilizes gpu (if available) for training.

**Prediction**
`python train.py input checkpoint --top_k [K]` - specifies to return the top `k` predictions by default it returns the top 5 predictions alongsides their corresponding probabilities.
`python train.py input checkpoint --category_names [CAT_TO_NAME]` - specifies whether to map numerical categories to labels and what mapping to use, the `cat_to_name` file should be in the `json` format, an example is provided alongsides the program.
`python train.py input checkpoint --gpu` - utilizes an available gpu for prediction.

> [!TIP]
> Make sure to use a gpu during the training process if available, otherwise you could spend a very long time staring at your screen - which isnt fun ):
