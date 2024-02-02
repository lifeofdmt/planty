from torch import optim, nn
from utility import program_parser, configure_device
from helper import load_dataloaders, train_model, save_model, load_model

parser = program_parser()
args = parser.parse_args()

# Store data directories
data_dir =  args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

# Create training and validation dataloaders
dataloaders, class_to_idx = load_dataloaders(train_dir, valid_dir)

# Configure training device
device = configure_device(args.gpu)

# Initialize model
model = load_model(args.arch) 

# Store input features of our classifier
if args.arch == "resnet50":
    input_features = model.fc.in_features
else:
    input_features = model.classifier[0].in_features

# Freeze pretrained network parameters
for param in model.parameters():
    param.requires_grad = False
    
# Create classifier
classifier = nn.Sequential(nn.Linear(input_features, 512),
                           nn.ReLU(),
                           nn.Dropout(0.3),
                           nn.Linear(512, 256),
                           nn.ReLU(),
                           nn.Dropout(0.3),
                           nn.Linear(256, 128),
                           nn.ReLU(),
                           nn.Dropout(0.3),
                           nn.Linear(128, 102),
                           nn.LogSoftmax(dim=1))

if args.arch == "resnet50":
    model.fc = classifier
    params = model.fc.parameters()
else:
    model.classifier = classifier
    params = model.classifier.parameters()
    
# Create optimizer
optimizer = optim.Adam(params, lr=args.learning_rate)
criterion = nn.NLLLoss()

model.to(device)
epochs = args.epochs

# Train model
model = train_model(model=model, optimizer=optimizer, epochs=epochs, criterion=criterion, 
                    device=device, dataloaders=dataloaders)

# Save model
if args.save_dir != None:
  save_model(save_dir=args.save_dir, model=model, optimizer=optimizer, input_features=input_features,
             class_to_idx=class_to_idx, arch=args.arch, epochs=epochs)
