import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ====================== MODEL ARCHITECTURE ======================
class TicTacToeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # New layer
        self.bn4 = nn.BatchNorm2d(256)  # New BN layer
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 3 * 3, 512)  # Adjusted for new layer
        self.fc2 = nn.Linear(512, 9)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))  # New layer
        x = x.view(-1, 256 * 3 * 3)  # Adjusted for new layer
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# ====================== DATA PREPROCESSING ======================
def preprocess(board):
    """
    Convert board state with dynamic player/opponent channels.
    Expects board values: 1 for player X, -1 for AI O, and 0 for empty.
    If board has 2 (for AI moves) instead of -1, convert it.
    """
    # Convert AI moves from 2 to -1:
    board = np.where(board == 2, -1, board)

    x_count = np.sum(board == 1)
    o_count = np.sum(board == -1)
    current_player = 1 if x_count == o_count else -1

    player_channel = (board == current_player).astype(float)
    opponent_channel = (board == -current_player).astype(float)
    empty_channel = (board == 0).astype(float)

    return np.stack([player_channel, opponent_channel, empty_channel], axis=0)


# ====================== HELPER FUNCTIONS ======================
def print_board(board):
    """Print the Tic-Tac-Toe board in a readable format."""
    symbols = {1: 'X', -1: 'O', 0: ' '}
    for row_idx in range(3):
        row_display = []
        for col_idx in range(3):
            cell = board[row_idx][col_idx]
            cell_display = symbols[cell]
            row_display.append(cell_display)
        print(" | ".join(row_display))
        if row_idx < 2:
            print("-" * 5)

def move_to_position(move):
    """Convert a move index (0-8) to a (row, col) position."""
    return (move // 3, move % 3)

def is_winning_move(board, player):
    """Check if the player has won."""
    for i in range(3):
        if all(board[i, :] == player) or all(board[:, i] == player):
            return True
    if all(board.diagonal() == player) or all(np.fliplr(board).diagonal() == player):
        return True
    return False

def is_board_full(board):
    """Check if the board is full (no empty spaces)."""
    return np.all(board != 0)

def get_valid_moves(board):
    """Get a list of valid moves (empty positions)."""
    return [move for move in range(9) if board[move // 3][move % 3] == 0]

def get_model_move(model, board):
    """Get the model's move for the current board state, ensuring it's valid."""
    model.eval()
    with torch.no_grad():
        test_input = torch.FloatTensor(preprocess(board)).unsqueeze(0)
        output = model(test_input)
        # Filter out invalid moves by setting their probabilities to -inf
        valid_moves = get_valid_moves(board)
        for move in range(9):
            if move not in valid_moves:
                output[0][move] = -float('inf')
        predicted_move = torch.argmax(output).item()
    return predicted_move

def get_human_move(board):
    """Get the human player's move."""
    while True:
        try:
            move = int(input("Enter your move (0-8): "))
            row, col = move_to_position(move)
            if 0 <= move <= 8 and board[row][col] == 0:
                return move
            else:
                print("Invalid move. Try again.")
        except ValueError:
            print("Invalid input. Enter a number between 0 and 8.")

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

# ====================== INTERACTIVE GAME ======================
def play_game(model, human_player=1):
    """Play a game of Tic-Tac-Toe against the model."""
    board = np.zeros((3, 3), dtype=int)  # Initialize empty board
    current_player = 1  # Human starts first if human_player=1

    print("Welcome to Tic-Tac-Toe!")
    print("You are playing as", 'X' if human_player == 1 else 'O')
    print("Enter moves as numbers 0-8, where 0 is the top-left corner and 8 is the bottom-right corner.")
    print_board(board)

    while True:
        if current_player == human_player:
            move = get_human_move(board)
        else:
            move = get_model_move(model, board)
            print(f"Model's move: {move}")

        row, col = move_to_position(move)
        board[row][col] = current_player
        print_board(board)

        if is_winning_move(board, current_player):
            print(f"{'You' if current_player == human_player else 'Model'} win!")
            break
        elif is_board_full(board):
            print("It's a draw!")
            break

        current_player *= -1  # Switch players
# ====================== EVALUATION FUNCTIONS ======================
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

# ====================== MAIN SCRIPT ======================
if __name__ == "__main__":
    # Load and clean data
    minimax_df = pd.read_csv("tic_tac_toe_records_minimax.csv")
    combined_rules_df = pd.read_csv("tic_tac_toe_records_combined_rules.csv")
    mcts_df = pd.read_csv("tic_tac_toe_records_mcts.csv")

    # Combine the datasets
    combined_df = pd.concat([combined_rules_df], ignore_index=True)

    # Remove duplicates
    combined_df = combined_df.drop_duplicates(subset=['board'])

    # Convert board strings to numerical arrays
    X, y = [], []
    for _, row in combined_df.iterrows():
        board_str = row['board'].ljust(9)
        board = np.array([list(board_str[i:i+3]) for i in range(0, 9, 3)])
        numerical_board = np.vectorize({'X': 1, 'O': -1, ' ': 0}.get)(board)
        X.append(numerical_board)
        y.append(row['decision'] - 1)  # Convert to 0-based index

    # Preprocess and convert to tensors
    X_processed = np.array([preprocess(board) for board in X])
    X_tensor = torch.FloatTensor(X_processed)
    y_tensor = torch.LongTensor(np.array(y))

    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)  # Increased batch size
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=128)

    # Initialize model and optimizer
    model = TicTacToeCNN()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)  # Adjusted learning rate and weight decay
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Learning rate scheduler

    # Load pre-trained model weights (if available)
    try:
        model.load_state_dict(torch.load("tic_tac_toe_model.pth"))
        print("Loaded pre-trained model weights.")
    except FileNotFoundError:
        print("No pre-trained model found. Training from scratch...")

    # Initialize lists to store losses
    train_losses = []
    val_losses = []

    # Training loop with early stopping
    num_epochs = 100  # Increased number of epochs
    best_val_loss = float('inf')
    patience = 10  # Increased patience for early stopping
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # Store losses for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")  # Save the best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step()  # Update learning rate

    # Load the best model
    model.load_state_dict(torch.load("best_model.pth"))

    # Evaluate the model
    evaluate_model(model, val_loader)

    # Plot learning curves
    plot_learning_curves(train_losses, val_losses)
    # Plot bias, variance, and total error
    plot_bias_variance(train_losses, val_losses)

    # Play the game
    play_game(model, human_player=-1)  # Set human_player to -1 if you want to play as O

