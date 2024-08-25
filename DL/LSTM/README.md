# LSTM Sine Wave Prediction with Streamlit Visualization

This project implements a Custom Long Short-Term Memory (LSTM) neural network to predict sine waves, now featuring an interactive Streamlit application for real-time visualization of the training process and predictions.

## Video Demonstration

[Watch the video](https://www.your-video-host.com/path/to/video.mp4)



Click the image above to watch a demonstration of the project in action.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Usage](#usage)
5. [Model Architecture](#model-architecture)
6. [Streamlit Application](#streamlit-application)
7. [Customization](#customization)
8. [Contributing](#contributing)
9. [License](#license)

## Project Overview

This project uses a custom implementation of an LSTM network to predict future values of a sine wave. It includes data generation, model training, and an interactive Streamlit application for visualizing results and experimenting with hyperparameters.

## Installation

To run this project, you need Python 3.6+ and the following libraries:

```bash
pip install numpy matplotlib streamlit jupyter
```

## Project Structure

- `LSTM.ipynb`: Jupyter notebook containing the LSTM class implementation.
- `LSTM.py`: Python file converted from LSTM.ipynb, containing the LSTM class.
- `utils.py`: Utility functions for data generation and preparation.
- `app.py`: Streamlit application for interactive visualization.
- `README.md`: This file, containing project documentation.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/lstm-sine-wave-prediction.git
   cd lstm-sine-wave-prediction
   ```

2. Convert the LSTM notebook to a Python file (if not already done):
   ```bash
   jupyter nbconvert --to python LSTM.ipynb
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

4. Open the provided URL in your web browser to interact with the application.

## Model Architecture

The LSTM model consists of:
- Input layer
- LSTM layer with customizable hidden units
- Output layer

The model uses various activation functions including sigmoid, tanh, and linear activations.

## Streamlit Application

The Streamlit app provides an interactive interface for:
- Adjusting hyperparameters (hidden size, learning rate, epochs, batch size)
- Visualizing the training process in real-time
- Displaying final predictions and performance metrics

Features of the Streamlit app:
- Live updating plots for predictions and training loss
- Progress bar for training
- Final evaluation on test data
- Display of performance metrics (MSE, RMSE, MAE)

## Customization

You can customize various aspects of the model through the Streamlit interface:
- Adjust the number of hidden units in the LSTM layer
- Modify the learning rate, number of epochs, or batch size
- Experiment with different optimizer types (implemented in the LSTM class)
