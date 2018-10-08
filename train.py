from utils import load_data,build_model, train_model, check_accuracy, save_checkpoint
import argparse
    
if __name__ == "__main__":


	parser = argparse.ArgumentParser(description='train.py')
	parser.add_argument('data_dir', default = 'flowers',  help = 'data directory to read the training, testing data from')
	parser.add_argument('--save_dir',  default = 'saved_model.pth', help = 'Directory to save the model')
	parser.add_argument('--arch',  default = 'vgg16', help = 'Architecture of the model')
	parser.add_argument('--learning_rate',  default = .001, help = 'Learning rate to train the model')
	parser.add_argument('--hidden_units',  default = 4096, help = 'No of hidden layers')
	parser.add_argument('--epochs', type = int,   default = 3, help = 'No of epochs to train the model')
	parser.add_argument('--gpu', default = 'gpu', help = 'Determine GPU vs CPU for the neural network')

	args = parser.parse_args()
	data_dir = args.data_dir
	save_dir = args.save_dir
	arch = args.arch
	learning_rate = args.learning_rate
	hidden_units = args.hidden_units
	epochs = args.epochs
	gpu = args.gpu

	if gpu == 'gpu':
	    is_gpu = True
	else:
	    is_gpu = False
	    
	image_datasets, dataloaders = load_data(data_dir)
	model = build_model(arch, hidden_units, is_gpu)
	model = train_model(model,dataloaders['training'], dataloaders['validation'], epochs, learning_rate, is_gpu)
	accuracy = check_accuracy(model, dataloaders['test'], is_gpu)
	print ("Accuracy: {:.2f}%".format(accuracy*100))
	save_checkpoint(model, hidden_units, epochs, image_datasets['training'], save_dir)
