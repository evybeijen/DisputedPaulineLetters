"""
Functions for creating and evaluating a bidirectional LSTM model for text classification.
"""

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score

# Data holder for the training data and size of the vocabulary
data = {
    'X': ...,  # Tokenized and padded input sequences
    'y': ...,  # Encoded labels corresponding to X
    'vocab_size': ...  # Computed vocab_size from tokenizer or tokenized input sequences
}

def create_model(embed_dim, vocab_size, lstm_units, dropout=0.2, learning_rate=0.001, num_classes=2, 
                 loss='sparse_categorical_crossentropy'):
    """
    Builds and compiles a bidirectional LSTM model for text classification.

    Args:
        embed_dim (int): Dimension of the embedding vectors.
        vocab_size (int): Size of the vocabulary (of the input sequences).
        lstm_units (int): Number of LSTM units (or hidden layer size).
        dropout (float): Dropout rate for the input vectors of the LSTM layer.
        learning_rate (float): Learning rate for the optimizer.
        num_classes (int): Number of output classes.
        loss (str): Loss function to use for model training. Defaults to 'sparse_categorical_crossentropy'.

    Returns:
        keras.Model: Compiled Keras model ready for training.
    """
    model = Sequential()

    # First layer: trainable embedding layer turning each tokenized word into a vector
    model.add(Embedding(input_dim=vocab_size, output_dim=embed_dim))

    # Second layer: bidirectional LSTM to capture context from both directions
    model.add(Bidirectional(LSTM(units=lstm_units, dropout=dropout)))

    # Final layer: output layer for classification, using the softmax activation function
    model.add(Dense(units=num_classes, activation='softmax'))

    # Compile the model with the Adam optimizer, the provided loss function, 
    # and accuracy as the evaluation metric
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    return model


def evaluate_model(embed_dim, vocab_size, lstm_units, dropout, learning_rate, epochs, batch_size, X, y, num_classes=2, 
                   loss='sparse_categorical_crossentropy'):
    """
    Evaluates a bidirectional LSTM model using five-fold cross-validation, given a set of hyperparameters and training data.

    Args:
        embed_dim (int): Dimension of the embedding vectors.
        vocab_size (int): Size of the vocabulary (of the input sequences).
        lstm_units (int): Number of LSTM units (or hidden layer size).
        dropout (float): Dropout rate for the input vectors of the LSTM layer.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        X (array-like): Tokenized and padded input sequences.
        y (array-like): Encoded labels corresponding to `X`.
        num_classes (int): Number of output classes. Defaults to 2.
        loss (str): Loss function to use for model training. Defaults to 'sparse_categorical_crossentropy'.

    Returns:
        float: Average cross-validation accuracy score.
    """
    
    # Create model with the given hyperparameters
    model = KerasClassifier(
        model=create_model,
        embed_dim=embed_dim, 
        vocab_size=vocab_size,
        lstm_units=lstm_units, 
        dropout=dropout,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        num_classes=num_classes,
        loss=loss,
        verbose=0
    )

    # Use five-fold cross-validation to evaluate model performance
    k_folds = KFold(n_splits=5, shuffle=True, random_state=43)
    scores = cross_val_score(model, X=X, y=y, cv=k_folds)

    # Return the average accuracy across folds
    return scores.mean()


def objective_function_tuning(embed_dim, lstm_units, dropout, learning_rate, epochs, batch_size):
    """
    Wrapper around the evaluate_model function for hyperparameter tuning (in our case, using Bayesian Optimization).

    Args:
        embed_dim (float): Dimension of the embedding vectors, to be cast to an integer.
        lstm_units (float): Number of LSTM units, to be cast to an integer.
        dropout (float): Dropout rate for the input vectors of the LSTM layer.
        learning_rate (float): Learning rate for the optimizer.
        epochs (float): Number of training epochs, to be cast to an integer.
        batch_size (float): Batch size for training, to be cast to an integer.

    Returns:
        float: The average cross-validation accuracy score from evaluate_model.

    Note:
        - `X` and `y` should be preprocessed and labeled training data.
        - The size of the vocabulary (`vocab_size`) can be retrieved from the tokenizer or 
          computed from the tokenized input sequences.
    """
    X = data['X']  # Access tokenized and padded input sequences
    y = data['y']  # Access encoded labels
    vocab_size = data['vocab_size']  # Access the computed size of the vocabulary

    # Cast the parameters that should be integers
    embed_dim = int(embed_dim)
    lstm_units = int(lstm_units)
    epochs = int(epochs)
    batch_size = int(batch_size)
    
    # Call the evaluate_model function and return the result
    return evaluate_model(embed_dim, vocab_size, lstm_units, dropout, learning_rate, epochs, batch_size, X, y)
