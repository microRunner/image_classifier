from utils import load_model, load_data, check_accuracy, predict_names
import argparse
import json
    
if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='predict.py')
	parser.add_argument('--image_path', type = str, default = 'flowers/test/101/image_07983.jpg',
	                    help = 'path of the image to be predicted')
	parser.add_argument('--checkpoint',  type = str, default = 'saved_model.pth', help = 'Directory to load the saved model')
	parser.add_argument('--top_k',  type = int, default = 5, help = 'No of classes and probabilites to be returned')
	parser.add_argument('--category_name', default = 'cat_to_name.json', help = 'Determine GPU vs CPU for the neural network')
	parser.add_argument('--gpu', type = str, default = 'gpu', help = 'Determine GPU vs CPU for the neural network')

	args = parser.parse_args()
	image_path = args.image_path
	checkpoint = args.checkpoint
	gpu = args.gpu
	top_k = args.top_k
	category_name = args.category_name

	if gpu == 'gpu':
	    is_gpu = True
	else:
	    is_gpu = False

	model = load_model(checkpoint, is_gpu)

	# to check if the model has been correctly built, and has significant accuracy
	# image_datasets, dataloaders = load_data('flowers')
	# check_accuracy(model, dataloaders['test'], is_gpu)

	with open(category_name, 'r') as f:
	    cat_to_name = json.load(f)
	    
	predict_names(model, image_path, cat_to_name, top_k)
