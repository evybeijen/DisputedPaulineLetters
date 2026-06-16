"""
Functions for creating and evaluating a bidirectional LSTM (text classification) model, intended for hyperparameter tuning.
"""

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score


def create_model(embed_dim, lstm_units, vocab_size, dropout=0.2, learning_rate=0.001, num_classes=2, 
                 loss='sparse_categorical_crossentropy'):
    """
    Builds and compiles a bidirectional LSTM model for text classification.

    Args:
        embed_dim (int): Dimension of the embedding vectors.
        lstm_units (int): Number of LSTM units (or hidden layer size).
        vocab_size (int): Size of the vocabulary (of the input sequences).
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


def evaluate_model(embed_dim, lstm_units, dropout, learning_rate, epochs, batch_size, vocab_size, X, y, num_classes=2, 
                   loss='sparse_categorical_crossentropy'):
    """
    Evaluates a bidirectional LSTM model using five-fold cross-validation, given a set of hyperparameters and training data.

    Args:
        embed_dim (int): Dimension of the embedding vectors.
        lstm_units (int): Number of LSTM units (or hidden layer size).
        dropout (float): Dropout rate for the input vectors of the LSTM layer.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        vocab_size (int): Size of the vocabulary (of the input sequences).
        X (array-like): Tokenized and padded input sequences.
        y (array-like): Encoded labels corresponding to `X`.
        num_classes (int): Number of output classes. Defaults to 2.
        loss (str): Loss function to use for model training. Defaults to 'sparse_categorical_crossentropy'.

    Returns:
        float: Average cross-validation accuracy score.
    """
    
    # Create model with the given hyperparameters
    model = KerasClassifier(
        build_fn=create_model,
        embed_dim=embed_dim, 
        lstm_units=lstm_units,
        vocab_size=vocab_size,
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
