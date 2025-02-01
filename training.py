import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Layer
from tensorflow.keras.layers import Reshape, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint

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
train_fingerprints = np.load('datasets/train_data/train_fingerprints.npy')
train_labels_label = np.load('datasets/train_data/train_labels_label.npy')
train_labels_patho = np.load('datasets/train_data/train_labels_patho.npy')
test_inputs = np.load('datasets/train_data/test_inputs.npy')
test_fingerprints = np.load('datasets/train_data/test_fingerprints.npy')
test_labels_label = np.load('datasets/train_data/test_labels_label.npy')
test_labels_patho = np.load('datasets/train_data/test_labels_patho.npy')

train_inputs = train_inputs.astype('float32')
train_fingerprints = train_fingerprints.astype('float32')
test_inputs = test_inputs.astype('float32')
test_fingerprints = test_fingerprints.astype('float32')

# Split training data into training and validation sets
train_inputs, val_inputs, train_labels_label, val_labels_label, train_labels_patho, val_labels_patho, train_fingerprints, val_fingerprints = train_test_split(
    train_inputs, train_labels_label, train_labels_patho, train_fingerprints, 
    test_size=0.1, random_state=42
)


def build_cascade_transformer_model_with_fingerprint(input_shape, fingerprint_shape, num_heads=4, ff_dim=128, task1_classes=2, task2_classes=2):
    """
    Build a multi-task learning model with transformer, cascade architecture, and fingerprint input.

    Args:
        input_shape (tuple): Shape of the primary input tensor (sequence_length, feature_dim).
        fingerprint_shape (int): Number of features in the fingerprint input.
        num_heads (int): Number of attention heads in the transformer.
        ff_dim (int): Feed-forward dimension in the transformer.
        task1_classes (int): Number of output classes for task 1.
        task2_classes (int): Number of output classes for task 2.

    Returns:
        Model: A compiled multi-task learning model.
    """
    # Primary input for sequence data
    sequence_input = Input(shape=input_shape, name="sequence_input")  # Shape: (sequence_length, feature_dim)

    # Fingerprint input
    fingerprint_input = Input(shape=(fingerprint_shape,), name="fingerprint_input")  # Shape: (fingerprint_dim,)
    fingerprint_dense = Dense(input_shape[1], activation="relu", name="fingerprint_dense")(fingerprint_input)  # Match feature_dim
    fingerprint_expanded = Reshape((1, input_shape[1]), name="fingerprint_expanded")(fingerprint_dense)  # Shape: (1, feature_dim)

    # Combine sequence input and fingerprint input along the sequence dimension
    combined_input = Concatenate(axis=1, name="combined_input")([sequence_input, fingerprint_expanded])  # Shape: (sequence_length + 1, feature_dim)

    # First Transformer Block
    x = TransformerBlock(embed_dim=input_shape[1], num_heads=num_heads, ff_dim=ff_dim)(combined_input)
    x = TransformerBlock(embed_dim=input_shape[1], num_heads=num_heads, ff_dim=ff_dim)(x)
    x = Flatten()(x)
    shared_dense = Dense(input_shape[1], activation='relu', name="shared_dense_1")(x)
    shared_dense_task1 = Dense(128, activation='relu', name="shared_dense_2")(shared_dense)
    shared_dense_task1 = Dense(32, activation='relu', name="shared_dense_3")(shared_dense_task1)
    shared_dropout = Dropout(0.3, name="shared_dropout_1")(shared_dense_task1)

    # Task 1 Output
    task1_output = Dense(task1_classes, activation='softmax', name="task1_output")(shared_dropout)

    # Cascade Task 1 Output to Task 2 Input
    task1_features = Dense(input_shape[1], activation='relu', name="task1_features")(task1_output)
    task1_features_expanded = tf.expand_dims(task1_features, axis=1)

    # Expand shared_dense and concatenate with task1_features
    shared_dense_expanded = tf.expand_dims(shared_dense, axis=1)
    cascade_input = tf.concat([shared_dense_expanded, task1_features_expanded], axis=1)  # Shape: (batch_size, 2, input_shape[1])

    # Task 2 Transformer Block
    task2_x = TransformerBlock(embed_dim=input_shape[1], num_heads=num_heads, ff_dim=ff_dim)(cascade_input)
    task2_x = Flatten()(task2_x)
    task2_x = Dense(128, activation='relu', name="task2_dense_1")(task2_x)
    task2_x = Dense(32, activation='relu', name="task2_dense_2")(task2_x)

    # Task 2 Output
    task2_output = Dense(task2_classes, activation='softmax', name="task2_output")(task2_x)

    # Build Model
    model = Model(inputs=[sequence_input, fingerprint_input], outputs=[task1_output, task2_output], name="cascade_transformer_with_fingerprint")
    return model
    
# Define input shapes
sequence_input_shape = (1024, 80)
fingerprint_input_shape = train_fingerprints.shape[1]  # 881

# Build model
model = build_cascade_transformer_model_with_fingerprint(
    input_shape=sequence_input_shape,
    fingerprint_shape=fingerprint_input_shape
)

model.summary()

weights_path = 'model/cascade_transformer_model_weights.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=weights_path,
    monitor='val_task1_output_accuracy',
    save_best_only=True,
    mode='max',
    save_weights_only=True, 
    verbose=1
)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=["categorical_crossentropy", "categorical_crossentropy"],
    metrics=["accuracy"]
)

# Train the model
history = model.fit(
    [train_inputs, train_fingerprints],
    [train_labels_label, train_labels_patho],
    validation_data=([val_inputs, val_fingerprints], [val_labels_label, val_labels_patho]),
    epochs=50,
    batch_size=32,
    verbose=1,
    callbacks=[model_checkpoint_callback] 
)

# Save the model weights
model.save_weights(weights_path)
print("Model weights saved successfully.")

test_loss, test_label_loss, test_patho_loss, test_label_acc, test_patho_acc = model.evaluate(
    [test_inputs, test_fingerprints],  # Include both sequence inputs and fingerprint inputs
    [test_labels_label, test_labels_patho],  # Outputs for both tasks
    verbose=1
)

print(f"Test Total Loss: {test_loss}")
print(f"Test Label Loss: {test_label_loss}")
print(f"Test Patho Loss: {test_patho_loss}")
print(f"Test Label Accuracy: {test_label_acc}")
print(f"Test Patho Accuracy: {test_patho_acc}")

test_predictions = model.predict([test_inputs, test_fingerprints])
task1_pred = np.argmax(test_predictions[0], axis=1)
task2_pred = np.argmax(test_predictions[1], axis=1)

test_df = pd.DataFrame()
test_df["task1_prediction"] = task1_pred
test_df["task2_prediction"] = task2_pred
test_df.to_csv("test_predictions.csv", index=False)

# Plot loss curves
print(history.history.keys())
plt.figure(figsize=(4, 3))
plt.plot(history.history['task1_output_loss'], label='Train Label Loss', color='#4B0082')
plt.plot(history.history['val_task1_output_loss'], label='Val Label Loss', color='#9370DB')
plt.plot(history.history['task2_output_loss'], label='Train Patho Loss', color='#006400')
plt.plot(history.history['val_task2_output_loss'], label='Val Patho Loss', color='#8FBC8F')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves for Multi-Task Learning')
plt.tight_layout()
plt.savefig('images/training-loss.png',dpi=300)
#plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, task_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(3, 3))  # Adjust the figure size here
    ax = plt.gca()  # Get the current axes
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Purples, ax=ax)  # Pass the axes explicitly
    plt.title(f"{task_name} \nConfusion Matrix")
    plt.tight_layout()
    plt.savefig('images/test-cm-' + task_name + '.png',dpi=300)
    #plt.show()

# Task 1 Confusion Matrix
plot_confusion_matrix(np.argmax(test_labels_label, axis=1), task1_pred, task_name="Resistance prediction")

# Task 2 Confusion Matrix
plot_confusion_matrix(np.argmax(test_labels_patho, axis=1), task2_pred, task_name="Pathogenic prediction")

# Function to plot ROC curve
def plot_roc_curve(y_true, y_score, task_name):
    
    fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(3, 3))
    plt.plot(fpr, tpr, color='#9370DB', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='#708090', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{task_name} \nROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('images/test-roc-' + task_name + '.png',dpi=300)
    #plt.show()

# Task 1 ROC Curve
plot_roc_curve(np.argmax(test_labels_label, axis=1), test_predictions[0], task_name="Resistance prediction")

# Task 2 ROC Curve
plot_roc_curve(np.argmax(test_labels_patho, axis=1), test_predictions[1], task_name="Pathogenic prediction")


