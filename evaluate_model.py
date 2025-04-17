from config import LIBRISPEECH_PATH, FEATURE_DIM, NUM_CLASSES
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchaudio.datasets import LIBRISPEECH
from torchaudio.transforms import Resample
from torch.utils.data import DataLoader
from model.feature_extractor import AudioFeatureExtractor
from model.calibration_net import FeatureCalibrationNet

BATCH_SIZE = 32
NUM_WORKERS = 4

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

def map_labels_to_classes(labels):
    unique_labels = torch.unique(labels)
    label_to_class = {label.item(): idx for idx, label in enumerate(unique_labels)}
    mapped_labels = torch.tensor([label_to_class[label.item()] for label in labels])
    return mapped_labels, len(unique_labels)

def evaluate_model():
    # Load dataset
    dataloader = load_dataset()

    # Extract data and labels from the DataLoader
    data, labels = next(iter(dataloader))

    # Map the labels to class indices
    mapped_labels, n_classes = map_labels_to_classes(labels)

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, mapped_labels, test_size=0.2, random_state=42)

    # Load the trained models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = AudioFeatureExtractor(FEATURE_DIM).to(device)
    calibration_net = FeatureCalibrationNet().to(device)
    calibration_net.load_state_dict(torch.load('model.pth'))
    calibration_net.eval()

    # Move data to device
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # Extract features first
    with torch.no_grad():
        # Reshape input for feature extraction (batch_size, sequence_length)
        X_test = X_test.squeeze(1)  # Remove channel dimension
        features = feature_extractor(X_test)
        
        # Pass features through calibration network
        predictions = calibration_net(features)
        probabilities = torch.nn.functional.softmax(predictions, dim=1)
        
        print("Features Shape:", features.shape)
        print("Raw Model Output Shape:", predictions.shape)
        print("Raw Model Output Sample:", predictions[0])
        print("Probabilities Shape:", probabilities.shape)
        print("Probabilities Sample:", probabilities[0])
        
        predicted_labels = torch.argmax(probabilities, dim=1).cpu().numpy()
        y_test = y_test.cpu().numpy()

    # Debugging: Print predictions and true labels
    print("Predictions:", predicted_labels)
    print("True Labels:", y_test)

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
