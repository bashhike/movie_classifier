#! /bin/python3

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.legacy import data
from data_loader import DataLoader
from preproecssing import PreprocessData
from models import BiLSTM
from argparse import ArgumentParser

import random
import spacy
import re


def model_accuracy(predict, y):
    """
    Helper function to calculate the accuracy.
    :param predict: Predictions from the model.
    :param y: True labels.
    :return: Accuracy metric.
    """
    true_predict = (predict.argmax(1) == y.argmax(1)).float()
    acc = true_predict.sum() / len(true_predict)
    return acc


def train_model(model, epochs, optimizer, loss_function, train_iterator, valid_iterator):
    """
    Helper function to train a model.
    :param model: Modeb object.
    :param epochs: Number of epochs to run.
    :param optimizer: Optimiser to use.
    :param loss_function: Loss criterion.
    :param train_iterator: Training data iterator.
    :param valid_iterator: Validation data iterator.
    :return: NA.
    """
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for i, batch in enumerate(train_iterator):
            (feature, batch_length), label = batch.overview, batch.genre
            batch_length = batch_length.to('cpu')
            label = label.float()
            optimizer.zero_grad()

            output = model(feature, batch_length)

            loss = loss_function(output, label)
            acc = model_accuracy(output, label)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += acc.item()
        print(
            f"Train:: Epoch: {epoch}, Loss: {train_loss / len(train_iterator)}, Accuracy: {train_acc / len(train_iterator)}")

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        for i, batch in enumerate(valid_iterator):
            (feature, batch_length), label = batch.overview, batch.genre
            batch_length = batch_length.to('cpu')
            label = label.float()

            output = model(feature, batch_length)
            loss = loss_function(output, label)
            acc = model_accuracy(output, label)

            val_loss += loss.item()
            val_acc += acc.item()

        print(
            f"Validation:: Epoch: {epoch}, Loss: {val_loss / len(valid_iterator)}, Accuracy: {val_acc / len(valid_iterator)}")
        print("")


def main():
    parser = ArgumentParser(description="Train the movie classifier!")
    parser.add_argument('--seed', type=int, metavar='int', required=False, help="Seed value", default=13)
    parser.add_argument('--batch', type=int, metavar='int', required=False, help="Define the batch size", default=128)
    parser.add_argument('--hidden', type=int, metavar='int', required=False, help="Define the hidden dimentions",
                        default=128)
    parser.add_argument('--lr', type=int, metavar='int', required=False, help="Define the learning rate", default=0.001)
    parser.add_argument('--dropout', type=int, metavar='int', required=False, help="Define the dropout prob",
                        default=0.2)
    parser.add_argument('--epochs', type=int, metavar='int', required=False, help="Number of epochs", default=10)
    parser.add_argument('--model_file', type=str, metavar='str', required=False, help='File to save model weights to.',
                        default='bilstm.pth')

    args = vars(parser.parse_args())

    seed = args['seed']
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dl = DataLoader('data')
    train_file = dl.get_training_data()
    allowed_genres = dl.get_allowed_genres()
    processor = PreprocessData(allowed_genres)
    genre_dict = processor.get_genre_dict()

    # Training specific variables.
    batch_size = args['batch']
    num_label = len(genre_dict)
    hidden_dim = args['hidden']
    learning_rate = args['lr']
    dropout = args['dropout']
    epochs = args['epochs']
    model_filename = args['model_file']

    train_iterator, valid_iterator = processor.get_iterators(train_file, batch_size, seed, device)
    embedding = processor.get_text().vocab.vectors.to(device)

    model = BiLSTM(
        embedding_dim=embedding.shape[1],
        hidden_dim=hidden_dim,
        label_size=num_label,
        batch_size=batch_size,
        embedding_weights=embedding,
        dropout=dropout
    )
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.BCELoss()

    train_model(model, epochs, optimizer, loss_function, train_iterator, valid_iterator)

    torch.save(model, 'weights/{}'.format(model_filename))
    torch.save(processor.get_text(), 'weights/{}'.format('text-' + model_filename))
    torch.save(processor.get_genre_dict(inverse=True), 'weights/{}'.format('dict-' + model_filename))
    print("Model weights saved in {}".format(model_filename))
    print("Text vocab saved in {}".format('text-' + model_filename))
    print("Label mappings saved in {}".format('dict-' + model_filename))


if __name__ == '__main__':
    main()
