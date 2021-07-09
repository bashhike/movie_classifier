#! /bin/python3

import os
import pandas as pd


class DataLoader(object):
    """
    Class to handle loading, downloading etc. of the datasets.
    """
    def __init__(self, data_dir):
        super(DataLoader, self).__init__()
        self.data_dir = data_dir
        self.raw_data_filename = 'movies_metadata.csv'
        self.processed_data_filename = 'movies_metadata_filtered.csv'
        self.raw_data_url = "https://google.com/robots.txt"

        # Variables specific to the data.
        self.allowed_genres = ['Animation', 'Comedy', 'Family', 'Adventure',
                               'Fantasy', 'Romance', 'Drama', 'Action', 'Crime',
                               'Thriller', 'Horror', 'History', 'Science Fiction',
                               'Mystery', 'War', 'Foreign', 'Music', 'Documentary',
                               'Western', 'TV Movie']

    def _download_data(self, file):
        """
        Function to download the moviefone dataset and extract movies_metadata.csv file.
        :param file: File to write the training data to.
        :return: NA
        """
        pass

    def _extract_genre(self, list_dict):
        if len(list_dict) > 0:
            return list_dict[0]['name']
        else:
            return 'unknown'

    def _process_data(self, raw_data, processed_data):
        """
        :param raw_data: Location of the raw data file.
        :param processed_data: Location to write the processed data.
        :return: Writes processed file to the specified processed_data_path.
        """
        df = pd.read_csv(raw_data, low_memory=False)
        df['genres'] = df['genres'].apply(eval)
        df['genre'] = df['genres'].apply(self._extract_genre)

        # Drop the columns where genre is not known.
        df = df.loc[df['genre'].isin(self.allowed_genres)]
        df.dropna()

        df = df[['title', 'overview', 'genre']]
        df.to_csv(processed_data, index=False)

    def get_training_data(self):
        """
        Helper function for getting the processed training data path.
        :return: Training data file path.
        """

        # Check if the processed data already exists
        processed_data_path = os.path.join(self.data_dir, self.processed_data_filename)
        if os.path.exists(processed_data_path):
            return processed_data_path

        # Fetch and generate the processed data if it doesn't exist.
        raw_data_path = os.path.join(self.data_dir, self.raw_data_filename)
        if os.path.exists(raw_data_path):
            self._process_data(raw_data_path, processed_data_path)
        else:
            self._download_data(raw_data_path)
            self._process_data(raw_data_path, processed_data_path)

        return processed_data_path

    def get_allowed_genres(self):
        """
        Helper function to fetch allowed genres.
        :return: List of allowed genres.
        """
        return self.allowed_genres
