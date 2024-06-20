import argparse
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.functional as F
import torchvision.models as models
import torch.optim as optim

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

#Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(from_file_path, to_device):
    
    checkpoint = torch.load(from_file_path, map_location=to_device)

    loaded_model = getModel(checkpoint['model_arch'] if 'model_arch' in checkpoint else 'vgg19')
    for param in loaded_model.parameters():
        param.requires_grad = False

    loaded_model.classifier = checkpoint['model_classifier']
    loaded_model.class_to_idx = checkpoint['model_class_to_idx']
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.to(to_device)
    

    #build optimizer
    loaded_optimizer = optim.Adam(loaded_model.classifier.parameters(), lr=float(checkpoint['optimizer_learning_rate'] if 'optimizer_learning_rate' in checkpoint else 0.001))
    loaded_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    loaded_epochs_completed = checkpoint['epochs_completed']
    
    #set criterion
    loaded_criterion = nn.NLLLoss()
    
    return loaded_model, loaded_criterion, loaded_optimizer, loaded_epochs_completed

def buildResizedCropped(image):
    RESIZE_MIN = 256
    CROP_WIDTH = 224
    
    crop_box_min = int((RESIZE_MIN - CROP_WIDTH) // 2) # in our case, 16
    crop_box_left = 0
    crop_box_upper = 0
    
    with Image.open(image) as img:
        w,h = RESIZE_MIN,RESIZE_MIN
        if img.width > img.height:
            w = int(img.width // (img.height / RESIZE_MIN))
            img_resized = img.resize((w,h))
            crop_box_left = int(img_resized.width // 2 - CROP_WIDTH // 2)
            crop_box_upper = crop_box_min
        else:
            h = int(img.height // (img.width / RESIZE_MIN))
            img_resized = img.resize((w,h))
            crop_box_left = crop_box_min
            crop_box_upper = int(img_resized.height // 2 - CROP_WIDTH // 2)
        
        crop_box = (crop_box_left,
                    crop_box_upper,
                    crop_box_left + CROP_WIDTH,
                    crop_box_upper + CROP_WIDTH)
        #print(crop_box)
        return img_resized.crop(crop_box)

#buildResizedCropped(test_dir + '/1/image_06743.jpg')

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    pil_image = buildResizedCropped(image)
    np_image = np.array(pil_image)
    np_image_mean = (np_image / 255.0) - np.array([0.485, 0.456, 0.406])
    np_image_div_by_std = np_image_mean / np.array([0.229, 0.224, 0.225])
    torch_compatible_np_image = np_image_div_by_std.transpose(2,0,1)
    torchImg = torch.from_numpy(torch_compatible_np_image)
    return torchImg

#process_image(test_dir + '/1/image_06743.jpg')

def getFloatTensorType(device):
    return torch.FloatTensor if device == "cpu" else torch.cuda.FloatTensor

def predict(image_path, model, on_device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    img = process_image(image_path)
    torchImg = img.type(getFloatTensorType(on_device)).unsqueeze(0)
    model.to(on_device)
    log_ps = model(torchImg)
    ps = torch.exp(log_ps)
    top_ps,top_class = ps.topk(topk, dim=1)
    top_ps_formatted = top_ps.data.cpu().numpy().squeeze()
    top_class_formatted = top_class.data.cpu().numpy().squeeze()
    
    if topk == 1:
        top_ps_formatted = [top_ps_formatted.tolist()]
        top_class_formatted = [top_class_formatted.tolist()]
    
    idx_to_class = {v:k for k, v in model.class_to_idx.items()} # reverse the categories dictionary
    top_class_ids = []
    for tc in top_class_formatted:
        top_class_ids.append(str(idx_to_class[tc]))
    
    #print(top_ps)
    #print(top_class)
    return top_ps_formatted,top_class_ids

#predict(test_dir + '/1/image_06743.jpg', model)

def main():
    parser = argparse.ArgumentParser(description='Predict a flower name from an image.')
    parser.add_argument('image_path', type=str,
                        help='A file path to an image of a flower')
    parser.add_argument('checkpoint_path', type=str,
                        help='A file path to a torchvision model checkpoint')
    parser.add_argument('--topk', type=int, default=1,
                        help='Return the top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='A file path to a custom mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for inference instead of CPU')

    args = parser.parse_args()
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    targeted_device = "cuda:0" if args.gpu else "cpu"
    loaded_model, loaded_criterion, loaded_optimizer, loaded_epochs_completed = load_checkpoint(args.checkpoint_path, targeted_device)
    
    top_ps, top_class = predict(args.image_path, loaded_model, targeted_device, args.topk)
    
    top_class_names = []
    for tc in top_class:
        top_class_names.append(cat_to_name[tc])
    
    for i in range(len(top_class)):
        print(top_class_names[i], top_ps[i])
    
if __name__ == '__main__':
    main()