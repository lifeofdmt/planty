from torchvision import models, datasets, transforms
import torch 
from torch import nn
from utility import process_image

def load_model(model):
    return eval(f"models.{model}(pretrained=True)")

def load_dataloaders(train_dir, valid_dir):
    """
    Creates and returns (train_dir, valid_dir) dataloaders 
    and class_to_idx mapping
    """
    # Define transforms for the training and validation sets
    data_transforms = [transforms.Compose([transforms.Resize(255), 
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomRotation(30),
                                           transforms.RandomHorizontalFlip(p=0.7),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                       transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])]

    # Load the training and validation
    image_datasets = [datasets.ImageFolder(train_dir, transform=data_transforms[0]), 
                      datasets.ImageFolder(valid_dir, transform=data_transforms[1])]

    # Define the dataloaders
    dataloaders = [torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True) for dataset in image_datasets]
    return (dataloaders, image_datasets[0].class_to_idx)

def train_model(model, optimizer, criterion, epochs, device, dataloaders):
    """
    Trains `model` with specified loss function and otpimizer
    for `epochs` epochs and returns a trained model
    
    """
    for e in range(epochs):
        training_loss = 0
        for images, labels in dataloaders[0]:
            # Move images and labels to default device
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad() # Clear accumulated gradients
            
            # Complete one pass through the network
            log_ps = model.forward(images)
            loss =  criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
        validation_loss = 0
        accuracy = 0

        model.eval()
        with torch.no_grad():
            for images, labels in dataloaders[1]:
                # Move images and labels to default device
                images, labels = images.to(device), labels.to(device)

                # Compute validation loss
                log_ps = model.forward(images)
                loss = criterion(log_ps, labels)
                validation_loss += loss.item()

                # Compute model's accuracy on validation set
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)

                equality = top_class.view(*labels.shape) == labels
                accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

        # Print model performance metrics
        print(f"Epoch {e + 1}")
        print(f"Training Loss: {training_loss/len(dataloaders[0]):.2f}")
        print(f"Validation Loss: {validation_loss/len(dataloaders[1]):.2f}")
        print(f"Accuracy: {(accuracy/len(dataloaders[1])) * 100:.2f}%")
        print()
        model.train()
    return model


def save_model(save_dir, model, optimizer, class_to_idx, input_features, arch, epochs):
    """
    Saves models parameter as `save_dir` on disk
    """
    if arch=='resnet50':
      state_dict = model.fc.state_dict()
    else:
      state_dict = model.classifier.state_dict()

    checkpoint = {'input_units': input_features, 'output_units': 102,
                  'hidden_units': [512, 256, 128],
                  'classifier_state_dict': state_dict,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'class_to_idx': class_to_idx,
                  'epochs': epochs,
                  'dropout': 0.3,
                  'arch': arch}
    torch.save(checkpoint, save_dir) # Save checkpoint to disk
    
    
def load_checkpoint(checkpoint):
    """
    Loads and returns a model from checkpoint
    file
    """
   
    checkpoint = torch.load(checkpoint)
    model = load_model(checkpoint['arch']) # Return a trained model
    
    # Freeze pretrained network parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Create classifier
    classifier = nn.Sequential(nn.Linear(checkpoint['input_units'], checkpoint['hidden_units'][0]),
                                nn.ReLU(),
                                nn.Dropout(checkpoint['dropout']),
                                nn.Linear(checkpoint['hidden_units'][0], checkpoint['hidden_units'][1]),
                                nn.ReLU(),
                                nn.Dropout(checkpoint['dropout']),
                                nn.Linear(checkpoint['hidden_units'][1], checkpoint['hidden_units'][2]),
                                nn.ReLU(),
                                nn.Dropout(checkpoint['dropout']),
                                nn.Linear(checkpoint['hidden_units'][2], checkpoint['output_units']),
                                nn.LogSoftmax(dim=1))
    
    if checkpoint['arch'] == 'resnet50':
        model.fc = classifier
        model.fc.load_state_dict(checkpoint['classifier_state_dict'])
    else:
        model.classifier = classifier
        model.classifier.load_state_dict(checkpoint['classifier_state_dict'])   
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def prediction(image_path, model_checkpoint, k, device, cat_to_name):
  """
  Make predictions and return `topk_ps` and `topk_classes`
  """
  model = load_checkpoint(model_checkpoint)
  model.eval()

  with torch.no_grad():
    image = (torch.from_numpy(process_image(image_path)))
    image = image.to(device)

    if device == 'cuda':
        log_ps = model(image.type(torch.cuda.FloatTensor).reshape((1,3,224,224)))
    else:
        log_ps = model(image.type(torch.FloatTensor).reshape((1,3,224,224)))        
    ps = torch.exp(log_ps)
    top_ps, top_classes = ps.topk(k, dim=1)
    top_ps = top_ps.tolist()[0]

    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    top_classes = top_classes.tolist()[0]
    top_classes = [idx_to_class[idx] for idx in top_classes]
    
    # Map class categories into labels
    if cat_to_name != None:
        top_classes = [cat_to_name[key] for key in top_classes]
  return (top_ps, top_classes)