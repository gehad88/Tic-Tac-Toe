import os
import torch
from flask import Flask, jsonify, request
import numpy as np
from model import TicTacToeCNN, preprocess
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Load the model once when the app starts
model = TicTacToeCNN()
model.load_state_dict(torch.load("best_model.pth"))
model.eval()  # Set the model to evaluation mode

# Helper functions (Ensure these are consistent with the Colab code)
def get_valid_moves(board):
    """Get a list of valid moves (empty positions)."""
    return [move for move in range(9) if board[move // 3][move % 3] == 0]

def get_model_move(board):
    """Get the model's move for the current board state, ensuring it's valid."""
    test_input = torch.FloatTensor(preprocess(board)).unsqueeze(0)
    with torch.no_grad():
        output = model(test_input)
        valid_moves = get_valid_moves(board)
        for move in range(9):
            if move not in valid_moves:
                output[0][move] = -float('inf')
        predicted_move = torch.argmax(output).item()
    return predicted_move

# ====================== ROUTES ======================

@app.route('/')
def home():
    return "Welcome to the Tic-Tac-Toe Model API!"

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to get the model's move."""
    data = request.get_json()  # Get the board state from the request
    board = np.array(data['board']).reshape(3, 3)  # Convert the board to a 3x3 numpy array
    
    # Get the model's move
    move = get_model_move(board)

    return jsonify({'move': move})

# ====================== EVALUATION FUNCTION (Optional) ======================

def evaluate_model(model, val_loader):
    """Evaluate the model on the validation set."""
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=[str(i) for i in range(9)]))

def plot_learning_curves(train_losses, val_losses):
    """Plot the training and validation loss curves."""
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.show()

def plot_bias_variance(train_losses, val_losses):
    """Plot the Bias-Variance Tradeoff."""
    epochs = range(1, len(train_losses) + 1)
    bias = np.array(train_losses)  # Approximation of bias as train loss
    variance = np.array(val_losses) - np.array(train_losses)  # Approximation of variance
    total_error = np.array(val_losses)  # Total error is the validation loss

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, bias, label='Bias (Train Loss)', color='blue', linestyle='--')
    plt.plot(epochs, variance, label='Variance (Validation - Train)', color='green', linestyle=':')
    plt.plot(epochs, total_error, label='Total Error (Validation Loss)', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Bias-Variance Tradeoff')
    plt.legend()
    plt.grid(True)
    plt.show()
    model.state_dict()

if __name__ == "__main__":
    app.run(debug=True)
