# Movie Classifier
Predict the genre of a movie given the title and description. 

Please use the included dockerfile to generate a docker image.
Inside the project directory, run: 
```bash
docker build -t movie_classifier . 
docker run -it movie_classifier
```
It'll take some time to download the packages and create a docker image. Once, that's done you can start classifying movies. 

Inside the container, you can run the application using: 
```bash
python movie_classifier.py --title <movie-title> --description <desc>
```
In order to run unit tests, run: 
```bash
python tests.py
```

You can also retrain the model by running the `train.py` file. 
Please check out the help section of the file for more details. 
You can access it by using: 
```bash
python train.py --help
```
---
### Details

- The dataset is derived from [The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset) on kaggle.
- The application uses a pytorch implementation of the BiLSTM model for prediction, since LSTM have been shown to perform well in sentence classification tasks. 
- GloVE 6b, 50 dimensional vectors are used in the embedding layer. 
- The dataset (`movies_metadata.csv`) has been processed to convert the problem from multi-label to a multi-class problem.
