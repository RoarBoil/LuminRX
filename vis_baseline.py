import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Layer

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import tensorflow.keras.backend as K

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.attention_weights = None  # Save attention weights here

    def call(self, inputs, training, return_attention=False):
        attn_output, attn_weights = self.att(inputs, inputs, return_attention_scores=True)
        self.attention_weights = attn_weights  # Save attention weights
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        if return_attention:
            return self.layernorm2(out1 + ffn_output), attn_weights
        return self.layernorm2(out1 + ffn_output)


# load datasets
train_inputs = np.load('datasets/train_data/train_inputs.npy')
train_labels_label = np.load('datasets/train_data/train_labels_label.npy')
train_labels_patho = np.load('datasets/train_data/train_labels_patho.npy')
test_inputs = np.load('datasets/train_data/test_inputs.npy')
test_labels_label = np.load('datasets/train_data/test_labels_label.npy')
test_labels_patho = np.load('datasets/train_data/test_labels_patho.npy')

train_inputs = train_inputs.astype('float32')
test_inputs = test_inputs.astype('float32')

# Split training data into training and validation sets
train_inputs, val_inputs, train_labels_label, val_labels_label, train_labels_patho, val_labels_patho = train_test_split(
    train_inputs, train_labels_label, train_labels_patho, test_size=0.1, random_state=42
)

# Build Multi-Task Learning Model with Transformer and Cascade State
def build_cascade_transformer_model(input_shape, num_heads=4, ff_dim=128, task1_classes=2, task2_classes=2):
    """
    Build a multi-task learning model with transformer and cascade architecture.

    Args:
        input_shape (tuple): Shape of the input tensor (sequence_length, feature_dim).
        num_heads (int): Number of attention heads in the transformer.
        ff_dim (int): Feed-forward dimension in the transformer.
        task1_classes (int): Number of output classes for task 1.
        task2_classes (int): Number of output classes for task 2.

    Returns:
        Model: A compiled multi-task learning model.
    """
    # Input layer
    inputs = Input(shape=input_shape, name="input_layer")

    # Transformer Block (shared layers)
    x = TransformerBlock(embed_dim=input_shape[1], num_heads=num_heads, ff_dim=ff_dim)(inputs)
    x = Flatten()(x)
    shared_dense = Dense(input_shape[1], activation='relu', name="shared_dense_1")(x)
    shared_dropout = Dropout(0.3, name="shared_dropout_1")(shared_dense)

    # Task 1 Output
    task1_output = Dense(task1_classes, activation='softmax', name="task1_output")(shared_dropout)

    # Cascade Task 1 Output to Task 2 Input
    task1_features = Dense(input_shape[1], activation='relu', name="task1_features")(task1_output)  # Process task1 output to match shared_dense dim
    task1_features_expanded = tf.expand_dims(task1_features, axis=1)  # Expand dims to match sequence

    # Expand shared_dense and concatenate with task1_features
    shared_dense_expanded = tf.expand_dims(shared_dense, axis=1)
    cascade_input = tf.concat([shared_dense_expanded, task1_features_expanded], axis=1)  # Shape: (batch_size, 2, input_shape[1])

    # Task 2 Transformer Block
    task2_x = TransformerBlock(embed_dim=input_shape[1], num_heads=num_heads, ff_dim=ff_dim)(cascade_input)
    task2_x = Flatten()(task2_x)
    task2_x = Dense(128, activation='relu', name="task2_dense_1")(task2_x)

    # Task 2 Output
    task2_output = Dense(task2_classes, activation='softmax', name="task2_output")(task2_x)

    # Build Model
    model = Model(inputs=inputs, outputs=[task1_output, task2_output], name="cascade_transformer_model")
    return model


    
# Compile the model
input_shape = (1024, 80)  # (sequence_length, feature_dim)
model = build_cascade_transformer_model(input_shape=input_shape)

model.summary()
model.load_weights('model/cascade_transformer_model_weights.h5')

# Extract attention weights for task2
# Modify the model to return attention weights
task2_transformer_block = model.get_layer("transformer_block_1")

@tf.function
def extract_attention_weights(inputs):
    _, attn_weights = task2_transformer_block(inputs, return_attention=True)
    return attn_weights

# Run the forward pass to compute attention weights
attention_weights = extract_attention_weights(test_inputs)
attention_weights = attention_weights.numpy()  # Convert to NumPy array

# Average attention weights across all heads
avg_attention_weights = np.mean(attention_weights, axis=1)  # Shape: (batch_size, seq_length, seq_length)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_attention_heatmap(attention_weights, normalize=True):
    """
    Plot heatmap of attention weights.

    Args:
        attention_weights (np.ndarray): Attention weights (batch_size, seq_length, seq_length).
        normalize (bool): Whether to normalize attention weights.
    """
    # Average across all samples
    avg_weights = np.mean(attention_weights, axis=0)  # Shape: (seq_length, seq_length)

    if normalize:
        avg_weights = avg_weights / np.max(avg_weights)  # Normalize for better visualization

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        avg_weights,
        cmap="viridis",
        cbar=True,
        xticklabels=["Original Input"] * avg_weights.shape[1] + ["Task 1 Features"] * avg_weights.shape[1],
        yticklabels=["Original Input"] * avg_weights.shape[0] + ["Task 1 Features"] * avg_weights.shape[0],
        square=True
    )
    plt.title("Task 2 Attention Weights")
    plt.xlabel("Attention on Inputs")
    plt.ylabel("Attention on Outputs")
    plt.xticks(
        ticks=[avg_weights.shape[1] // 4, avg_weights.shape[1] * 3 // 4],
        labels=["Original Input", "Task 1 Features"],
        fontsize=10
    )
    plt.yticks(
        ticks=[avg_weights.shape[0] // 4, avg_weights.shape[0] * 3 // 4],
        labels=["Original Input", "Task 1 Features"],
        fontsize=10
    )
    plt.show()


def plot_attention_distributions(attention_weights):
    """
    Plot attention weight distributions for original inputs and Task 1 features.

    Args:
        attention_weights (np.ndarray): Attention weights (batch_size, seq_length, seq_length).
    """
    # Split into Original Input and Task 1 Features
    seq_length = attention_weights.shape[1]
    mid_point = seq_length // 2

    original_attention = attention_weights[:, :, :mid_point].reshape(-1)  # Original input attention
    task1_attention = attention_weights[:, :, mid_point:].reshape(-1)  # Task 1 feature attention

    # Combine data for visualization
    data = {
        "Attention Weights": np.concatenate([original_attention, task1_attention]),
        "Input Type": ["Original Input"] * len(original_attention) + ["Task 1 Features"] * len(task1_attention),
    }

    import pandas as pd
    attention_df = pd.DataFrame(data)

    # Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Input Type", y="Attention Weights", data=attention_df, palette="Set2")
    plt.title("Task 2 Attention Weight Distribution")
    plt.ylabel("Attention Weights (Normalized)")
    plt.xlabel("Input Source")
    plt.show()

    # Violin Plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Input Type", y="Attention Weights", data=attention_df, palette="Set2", cut=0)
    plt.title("Task 2 Attention Weight Distribution (Violin Plot)")
    plt.ylabel("Attention Weights (Normalized)")
    plt.xlabel("Input Source")
    plt.show()

# Normalize attention weights
attention_weights = attention_weights / np.max(attention_weights)

# Call heatmap visualization
plot_attention_heatmap(avg_attention_weights, normalize=True)

# Call distribution visualization
plot_attention_distributions(attention_weights)
