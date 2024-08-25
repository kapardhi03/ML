import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from LSTM_ import LSTM, generate_sine_wave, prepare_data

st.set_page_config(page_title="LSTM Sine Wave Prediction", layout="wide")

st.sidebar.header("Hyperparameters")
hidden_size = st.sidebar.slider("Hidden Size", 10, 100, 50)
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, format="%.3f")
epochs = st.sidebar.slider("Epochs", 10, 500, 100)
batch_size = st.sidebar.slider("Batch Size", 8, 64, 32)

st.title("LSTM Sine Wave Prediction")

@st.cache_data
def get_data():
    samples = 1000
    sine_wave = generate_sine_wave(samples)
    n_steps = 50
    X, y = prepare_data(sine_wave, n_steps)
    train_size = int(0.8 * len(X))
    return X[:train_size], y[:train_size], X[train_size:], y[train_size:]

X_train, y_train, X_test, y_test = get_data()

input_size = 1
output_size = 1
model = LSTM(input_size, hidden_size, output_size, optimizer='adam')

def train_model(model, X, y, epochs, learning_rate, batch_size):
    losses = []
    predictions = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    plt.subplots_adjust(hspace=0.5)
    line1, = ax1.plot([], [], 'b-', label='Actual')
    line2, = ax1.plot([], [], 'r--', label='Predicted')
    ax1.set_title("Sine Wave Prediction")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Value")
    ax1.legend()
    
    loss_line, = ax2.plot([], [], 'g-')
    ax2.set_title("Training Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_yscale('log')
    
    plot_placeholder = st.empty()

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_predictions = []
        
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            batch_loss = 0
            batch_predictions = []
            for j in range(len(batch_X)):
                output, _ = model.forward(batch_X[j])
                loss = np.mean((output - batch_y[j])**2)
                batch_loss += loss
                batch_predictions.append(output)
                
                d_y = 2 * (output - batch_y[j]) / len(batch_X)
                model.backward(d_y, learning_rate=learning_rate)
            
            epoch_loss += batch_loss / len(batch_X)
            epoch_predictions.extend(batch_predictions)
        
        average_loss = epoch_loss / (len(X) // batch_size)
        losses.append(average_loss)
        predictions.append(epoch_predictions)
        
        # Update progress
        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f"Epoch {epoch+1}/{epochs}, Loss: {average_loss:.4f}")
        
        # Update plots
        line1.set_data(range(len(y)), y)
        line2.set_data(range(len(epoch_predictions)), epoch_predictions)
        ax1.relim()
        ax1.autoscale_view()
        
        loss_line.set_data(range(len(losses)), losses)
        ax2.relim()
        ax2.autoscale_view()
        
        plot_placeholder.pyplot(fig)
    
    return losses, predictions

if st.button("Train Model"):
    losses, predictions = train_model(model, X_train, y_train, epochs, learning_rate, batch_size)
    
    test_predictions = []
    for i in range(len(X_test)):
        output, _ = model.forward(X_test[i])
        test_predictions.append(output)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    plt.subplots_adjust(hspace=0.5)
    
    ax1.plot(y_test, 'b-', label='Actual')
    ax1.plot(test_predictions, 'r--', label='Predicted')
    ax1.set_title("Final Sine Wave Prediction")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Value")
    ax1.legend()
    
    ax2.plot(losses, 'g-')
    ax2.set_title("Training Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_yscale('log')
    
    st.pyplot(fig)
    
    mse = np.mean((np.array(test_predictions) - y_test)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(np.array(test_predictions) - y_test))
    st.write(f"Mean Squared Error: {mse:.4f}")
    st.write(f"Root Mean Squared Error: {rmse:.4f}")
    st.write(f"Mean Absolute Error: {mae:.4f}")

st.markdown("""
## Instructions
1. Adjust the hyperparameters in the sidebar as desired.
2. Click the "Train Model" button to start training.
3. Watch the live updates of the model's predictions and training loss.
4. After training, view the final prediction results and performance metrics.
""")