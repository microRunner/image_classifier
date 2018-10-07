import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image


def load_data(data_dir):
    """Load the training, validation and the test data.
    
    Args:
        data_dir: The data sirectory which will host the training, validation, and the test data.
    Returns:
        dataloaders: A dictionary with PyTorch Dataloaders for training, test, and validation data
    """
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test' 
    data_transforms = {
    'training' : transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])]) ,

    'val|test' : transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    }


    # TODO: Load the datasets with ImageFolder
    image_datasets = {
    'training' : datasets.ImageFolder(train_dir, transform=data_transforms['training']) ,
    'validation': datasets.ImageFolder(valid_dir, transform=data_transforms['val|test']) ,
    'test' : datasets.ImageFolder(test_dir, transform=data_transforms['val|test'])
    }


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
    'training' : torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True) ,
    'validation' : torch.utils.data.DataLoader(image_datasets['validation'], batch_size=32) ,
    'test' : torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
    }
    
    return image_datasets, dataloaders

def build_model(arch, hidden_layers, is_gpu):
    """Build a pretrained model with custom classifier. 
    Model will have two hidden layers,
    and support three kind of architectures.
    
    Args:
        arch: Architecture 
        hidden_layers: Hidden Layer
        is_gpu: Boolean flag for the use of GPUs
    Returns:
        The requisite model with pretrained features, and adjusted classifier
    """
    if arch == 'vgg16':
        model = models.vgg16(pretrained = True)
        input_layer = 25088
    elif arch == 'densenet161' :
        model = models.densenet161(pretrained = True)
        input_layer = 2208
    elif arch == 'alexnet':
        model = models.alexnet(pretrained = True)
        input_layer = 9216
    else: 
        raise ValueError("The arch should be in ['vgg16', ''densenet161', alexnet']")
        
    output_layers = 1002 # No of clases    

    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_layer, hidden_layers)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_layers, output_layers)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    if is_gpu:
        model.cuda()
    return model

def train_model(model, training_data, epochs, learning_rate, is_gpu):
    """Train the model
    Args:
        model: NN Model to be trained
        training_data: Trainig data for the model to be trained
        epochs: no of times the model will go over all the images
        learning_rate: learning rate
        is_gpu: Boolean flag to indicate if GPU is to be used or not
    
    
    """
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    steps = 0

    # change to cuda
    if is_gpu:
        model.to('cuda')

    for e in range(epochs):
        running_loss = 0
        no_of_steps_error = 0
        for _, (inputs, labels) in enumerate(training_data ):
            no_of_steps_error += 1
            steps += 1
            
            if is_gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print("Epoch: {}/{}... ".format(e+1, epochs),
              "Loss: {:.4f}".format(running_loss/no_of_steps_error))
    return model

def check_accuracy(nn_model, testing_data , is_gpu ):
    """Prints accuracy for a model for a certain dataset_type
    
    Args:
        nn_model: Particular Model
        testing_data : The data on which accuracy is to be tested
        is_gpu: Boolean flag to indicate if GPU is to be used or not

    Returns:
        accuracy: Accuracy of the model 
    """
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testing_data:
            images, labels = data
            if is_gpu:
                images, labels = images.to('cuda'), labels.to('cuda')
            outputs = nn_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy =  correct / total    
    print('Accuracy: %d %%' % (100 * correct / total))
    return accuracy

def save_checkpoint(model, hidden_layers , epochs , training_data , save_dir):
    """Save the model to the directory

    Args:
        mode: nn_model to be saved
        hidden_laters: no of hidden_layers in the model
        epochs: no of times the model was trained on the data
        training_data: training data
        save_dir: the directory where the model was saved
    Returns:
        None    
    """

    
    model.class_to_idx = training_data.class_to_idx
    torch.save({
                'hidden_layers':hidden_layers,
                'arch': 'vgg16',
                'no_of_epochs': epochs,
                'optimizer': 'adam',
                'class_to_idx':model.class_to_idx,
                'state_dict':model.state_dict()},
                 save_dir)
    
def load_model(path, is_gpu):
    """Recall the model's characteristics, and then re-build it
    
    Args:
        path: Path where model was saved
    Returns:
        loaded_model: Re-built model
    
    """
    saved_model = torch.load(path)
    arch = saved_model['arch']
    hidden_layers = saved_model['hidden_layers']
    loaded_model = build_model(arch , hidden_layers, is_gpu)
    loaded_model.class_to_idx = saved_model['class_to_idx']
    loaded_model.load_state_dict(saved_model['state_dict'])
    return loaded_model    

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    transform = transforms.Compose([transforms.Resize(256),
                       transforms.CenterCrop(224),
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], 
                                            [0.229, 0.224, 0.225])])
    transformed_img = transform(img)
    
    return transformed_img.numpy() 

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    
    Args:
        image_path:image path
        model: model used for inference
        topk: No of probabilities and classes to be retreived
    '''
    image = process_image(image_path)
    image = torch.from_numpy(image)
    # https://discuss.pytorch.org/t/converting-numpy-array-to-tensor-on-gpu/19423
    # Forum used to uncover the 'magical' function unsqueeze
    image = image.unsqueeze_(0).float()
    image = image.to('cuda')
    output = model.forward(image)
    probability = F.softmax(output.data,dim=1)
    return probability.topk(topk)

def predict_names(model, image_path, cat_to_name, topk):
    """Does an inference check for a particular flower image
    
    Args:
        model: model to be used for inference
        image_path: Image path
        cat_to_name: category to name dictionary
        topk: No of probabilities and classes to be retreived

    Returns:
        None
    
    """
    prob, cat = predict(image_path, model, topk)
    prob = prob.cpu().numpy()[0]
    cat = cat.cpu().numpy()[0]
    cat_name = []
    for category in cat:
        cat_name.append(cat_to_name[str(category)])
    for i in range(topk):
        print("The image is {} with probability {:.2f}%.".format(cat_name[i], prob[i]*100))
