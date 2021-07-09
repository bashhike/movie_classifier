#! /bin/python3

import numpy as np
import torch
from argparse import ArgumentParser


def inference(title, description, modelpath):
    """
    Function for performing inference of the model.
    :param title: Title of the movie.
    :param description: The description of the movie.
    :param modelpath: Model weights to load.
    :return: Model output in dictionary format.
    """
    # No gradient computation required during inference.
    with torch.no_grad():
        model_weights_file = modelpath
        sample = description
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model weights and vocab from disk. Map location to make it training device agnostic.
        model = torch.load('weights/{}'.format(model_weights_file), map_location=device)
        text = torch.load('weights/{}'.format('text-' + model_weights_file), map_location=device)
        label_dict = torch.load('weights/{}'.format('dict-' + model_weights_file), map_location=device)

        # Set model to eval mode to disable dropout and all that.
        model.eval()

        # Process the sample description to pass on to the model.
        sample_tokens = text.preprocess(sample)
        sample_vectors = np.asarray([[text.vocab.stoi[x] for x in sample_tokens]])
        sample_tensor = torch.LongTensor(sample_vectors).T
        sample_tensor = sample_tensor.to(device)
        len_tensor = torch.LongTensor([len(sample_vectors)])

        prediction = model(sample_tensor, len_tensor)
        pred_class = prediction.argmax(1).item()

        # Format the output.
        ret = dict()
        ret['title'] = title
        ret['description'] = description
        ret['genre'] = label_dict[pred_class]

        return ret


def main():
    parser = ArgumentParser(description="Movie classifier")
    parser.add_argument('--title', type=str, metavar='str', required=True, help="Title of the movie")
    parser.add_argument('--description', type=str, metavar='str', required=True, help="Overview of the movie")
    parser.add_argument('--modelpath', type=str, metavar='path', required=False,
                        help='Path to the saved model weights.', default='bilstm.pth')

    args = vars(parser.parse_args())

    ret = inference(args['title'], args['description'], args['modelpath'])
    print(ret)

if __name__ == '__main__':
    main()
