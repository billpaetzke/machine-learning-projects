import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from workspace_utils import active_session
import json

FINAL_CROP_WIDTH = 224

def buildArgParser():
    parser = argparse.ArgumentParser(description='Train a new network on a data set and save the model as a checkpoint.')
    parser.add_argument('data_dir', type=str,
                        help='A directory that contains the data set')
    parser.add_argument('--save_dir', type=str,
                        help='Set directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg19',
                        help='Choose architecture from vgg13, vgg16, vgg19, or densenet121')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Choose learning rate')
    parser.add_argument('--hidden_units', type=int,
                        help='Choose number of hidden layers')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Choose number of epochs')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for training instead of CPU')
    return parser

def buildNormalizeTransform():
    return transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

def composeTrainingTransform():
    return transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(FINAL_CROP_WIDTH), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        buildNormalizeTransform()])

def composeCleanTransform():
    return transforms.Compose([
        transforms.Resize(255), 
        transforms.CenterCrop(FINAL_CROP_WIDTH),
        transforms.ToTensor(),
        buildNormalizeTransform()])
    
def getFloatTensorType(device):
    return torch.FloatTensor if "cpu" else torch.cuda.FloatTensor

def getModel(from_arch):
    if from_arch == 'vgg13':
        return models.vgg13(pretrained=True)
    elif from_arch == 'vgg16':
        return models.vgg16(pretrained=True)
    elif from_arch == 'vgg19':
        return models.vgg19(pretrained=True)
    elif from_arch == 'densenet121':
        return models.densenet121(pretrained=True)
    else:
        return models.vgg19(pretrained=True)

def getFeatureCount(from_arch):
    if from_arch in ['vgg13','vgg16','vgg19']:
        return 25088
    elif from_arch == 'densenet121':
        return 1024
    else:
        return 25088

def getHiddenUnitsCountDefault(from_arch):
    if from_arch in ['vgg13','vgg16','vgg19']:
        return 1024
    elif from_arch == 'densenet121':
        return 512
    else:
        return 256
    
def save_checkpoint(model, model_arch, optimizer, optimizer_learning_rate, epochs_completed, class_to_idx, to_file_path):
    model.cpu()
    checkpoint = {
        'model_arch': model_arch,
        'model_classifier': model.classifier,
        'model_state_dict': model.state_dict(),
        'optimizer_learning_rate': optimizer_learning_rate,
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs_completed': epochs_completed,
        'model_class_to_idx': class_to_idx
    }
    torch.save(checkpoint, to_file_path)
    
    
def main():
    
    parser = buildArgParser()
    args = parser.parse_args()
    
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, composeTrainingTransform())
    valid_data = datasets.ImageFolder(valid_dir, composeCleanTransform())
    test_data = datasets.ImageFolder(test_dir, composeCleanTransform())

    # Using the image datasets and the trainforms, define the dataloaders
    BATCH_SIZE = 32
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)

    #initialize training #########################

    #set status
    epochs_completed = 0

    #set expectation
    epochs_target = args.epochs # train until

    #duration of this training session
    epochs = epochs_target - epochs_completed

    #set device
    device = "cuda" if args.gpu else "cpu"

    #build model
    training_model = getModel(args.arch)
    for param in training_model.parameters():
        param.requires_grad = False    
    
    feature_count = getFeatureCount(args.arch)
    hidden_units_count = args.hidden_units if args.hidden_units is not None else getHiddenUnitsCountDefault(args.arch)
    training_model.classifier = nn.Sequential(nn.Linear(feature_count, hidden_units_count),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(hidden_units_count, 102),
                                nn.LogSoftmax(dim=1))

    training_model.to(device)

    #build optimizer
    optimizer = optim.Adam(training_model.classifier.parameters(), lr=args.learning_rate)

    #set criterion
    criterion = nn.NLLLoss()
    
    #train ##################

    #print("Epochs completed:{} targeted:{} training:{} ...".format(epochs_completed, epochs_target, epochs))
    training_losses, validation_losses = [], []
    with active_session(): #to keep vm workspace alive for more than 30 minutes
        for e in range(epochs):

            running_loss = 0

            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                log_ps = training_model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epochs_completed += 1

            validation_loss = 0
            validation_accuracy = 0
            floatTensorType = getFloatTensorType(device)

            training_model.eval()
            with torch.no_grad():

                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)

                    log_ps = training_model(images)
                    loss = criterion(log_ps, labels)
                    validation_loss += loss.item()
                    ps = torch.exp(log_ps)
                    top_ps, top_class = ps.topk(1, dim=1)
                    correct = top_class == labels.view(*top_class.shape)
                    accuracy = torch.mean(correct.type(floatTensorType))
                    validation_accuracy += accuracy.item()

            training_model.train() 

            training_losses.append(running_loss/len(trainloader))
            validation_losses.append(validation_loss/len(validloader))
            print('E:{} Training Loss:{:.3f} Validation Loss:{:.3f} Accuracy:{:.3f}'.format(
                epochs_completed,
                running_loss/len(trainloader),
                validation_loss/len(validloader),    
                (validation_accuracy/len(validloader))*100))

    save_dir = args.save_dir+'/' if args.save_dir is not None else ''
    save_checkpoint(training_model, args.arch, optimizer, args.learning_rate, epochs_completed, 
                    train_data.class_to_idx, 
                    save_dir + 'checkpoint_'+args.arch+'h'+str(hidden_units_count)+'e'+str(epochs_completed)+'.pth')
    
if __name__ == '__main__':
    main()