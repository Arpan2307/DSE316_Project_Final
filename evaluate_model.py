from config import LIBRISPEECH_PATH, FEATURE_DIM, NUM_CLASSES
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchaudio.datasets import LIBRISPEECH
from torchaudio.transforms import Resample
from torch.utils.data import DataLoader
from model.feature_extractor import AudioFeatureExtractor
from torch import nn

BATCH_SIZE = 32
NUM_WORKERS = 4

# Adjust the model architecture to match the saved state_dict
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, FEATURE_DIM),  # Layer 0
            nn.ReLU(),
            nn.Linear(FEATURE_DIM, FEATURE_DIM)  # Layer 2
        )

    def forward(self, x):
        return self.net(x)

def load_dataset():
    # Define dataset path and parameters
    os.makedirs(LIBRISPEECH_PATH, exist_ok=True)
    dataset = LIBRISPEECH(LIBRISPEECH_PATH, url="train-clean-100", download=True)
    resampler = Resample(orig_freq=16000, new_freq=16000)

    def collate_fn(batch):
        waveforms, labels = [], []

        # Find the maximum length of waveforms in the batch
        max_length = max(waveform.shape[1] for waveform, _, _, *_ in batch)

        for waveform, _, speaker_id, *_ in batch:
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Pad the waveform to the maximum length
            padding = max_length - waveform.shape[1]
            padded_waveform = torch.nn.functional.pad(waveform, (0, padding))

            waveforms.append(padded_waveform)
            labels.append(hash(speaker_id) % (10 ** 8))  # Use a hash of the speaker_id for unique numeric labels

        return torch.cat(waveforms), torch.tensor(labels)

    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)

def evaluate_model():
    # Load dataset
    dataloader = load_dataset()

    # Extract data and labels from the DataLoader
    data, labels = next(iter(dataloader))

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Load the trained model
    model = Model()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    # Ensure the input data is resized to match the model's expected input dimensions
    X_test = X_test.view(X_test.size(0), -1)[:,:FEATURE_DIM]

    # Fix tensor warning by using clone().detach()
    X_test = X_test.clone().detach()

    # Perform evaluation on the test set
    with torch.no_grad():
        predictions = model(X_test.float())
        predicted_labels = torch.argmax(predictions, dim=1).numpy()

    # Debugging: Print predictions and true labels
    print("Predictions:", predicted_labels)
    print("True Labels:", y_test.numpy())

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predicted_labels)
    precision = precision_score(y_test, predicted_labels, average='weighted')
    recall = recall_score(y_test, predicted_labels, average='weighted')
    f1 = f1_score(y_test, predicted_labels, average='weighted')

    # Print evaluation metrics
    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

if __name__ == "__main__":
    evaluate_model()
