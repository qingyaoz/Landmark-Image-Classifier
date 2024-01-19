# Landmark-Image-Classifier
This project is primarily a personal learning exercise, maintained on GitHub for documentation.

## Overview
This project aims to develop a classifier using supervised deep learning techniques for image data to distinguishe images of two famous landmark: the Pantheon and the Hofburg Imperial Palace.

## Techniques
- **Data Preprocessing**: Refine the dataset by normalizing the data to make it more suitable for effective model training.
- **Convolutional Neural Network**: Utilize CNN to build model for image classification.
- **Grad-CAM**: Visualize which regions of an image contribute most to the final classification.
- **Transfer Learning**: Explore supervised pretraining to augment classification abilities..
- **Data Augmentation**: Experiment with data augmentation to enrich training data and improve the robustness of models. 

## Dataset
The dataset contains 5,152 PNG images of 10 different landmarks, including the Pantheon, the Hofburg Imperial Palace, the Colosseum, Rialto Bridge, Museu Nacional d’Art de Catalunya, Petronas Towers, Berlin Cathedral, Hagia Sophia, Gaud ́ı Casa Batllo ́ in Barcelona, and St Stephen’s Cathedral in Vienna.
- Each image is 3 × 64 × 64 with 3 color channels (RGB).
- The dataset is divided into training, validation, test, and challenge sets.
- `data/landmarks.csv` contains image labels and data partition information.

To prepare the data, unzip `data/images.zip` and ensure images are stored under `data/images/`.


## Development Environment
1. imageio (2.31.1)
2. matplotlib (3.7.1)
3. numpy (1.25.0)
4. pandas (2.0.2)
5. Pillow (9.5.0)
6. scikit-learn (1.2.2)
7. scipy (1.10.1)
8. torch (2.0.1)
9. torchvision (0.15.2)

To install the correct versions of the required packages, run the command `pip install -r requirements.txt` in your virtual environment.

## Skeleton Code
- **Executables**: Scripts for training, data augmentation, and predictions.
  - `augment_data.py`: Create an augmented dataset.
  - `confusion_matrix.py`: Generate confusion matrix graphs.
  - `matrix_challenge.py`: Customized version of `confusion_matrix.py` for challenge data.
  - `dataset.py`: Class wrapper for interfacing with the dataset of landmark images.
  - `predict_challenge.py`: Runs the challenge model inference on the test dataset and saves the
    predictions to disk.
  - `test_cnn.py`: Test trained CNN from train_cnn.py on the heldout test data. Load the trained CNN model from a saved checkpoint and evaulates using accuracy and AUROC metrics.
  - `test_challenge.py`: Customized version of `test_cnn.py` for challenge model.
  - `train_cnn.py`: Train a convolutional neural network to classify images. Periodically output training information, and save model checkpoints.
  - `train_challenge.py`: Customized version of `train_cnn.py` for the heldout images.
  - `train_source.py`: Customized version of `train_cnn.py` for source task of transfer learning.
  - `train_target.py`: Customized version of `train_cnn.py` for transfer learning model.
  - `visualize_cnn.py`: Grad-CAM Visualization, generates a heat map on top of the original image for 20 sample images in the dataset.
  - `visualize_data.py`: Open up a window displaying randomly selected training images.
- **Models**: Constructs a pytorch model for a convolutional neural network.
  - `model/challenge.py`: Neural network for challenge prediction.
  - `model/source.py`: Neural network for source task of transfer learning.
  - `model/target.py`: Neural network for transfer learning model.
- **Checkpoints**: Directory for model checkpoints.
  - `checkpoints/`
- **Data**: Image files and metadata.
  - `data/landmarks.csv`
  - `data/images.zip` (Unzip this folder to use the images)
- **Utilities**: Configuration and utility scripts.
  - `config.json`
  - `rng_control.py`
  - `train_common.py`: Helper file for common training functions.
  - `utils.py`: Utility functions
  - `requirements.txt`: Scripts to install all necessary Python packages.
- `prediction.csv`: My final prediction.
- `challenge_output.txt`: Record for outputs of different models and hyperparameters.
