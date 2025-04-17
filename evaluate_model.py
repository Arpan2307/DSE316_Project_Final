from config import LIBRISPEECH_PATH
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchaudio.datasets import LIBRISPEECH
from torchaudio.transforms import Resample
from torch.utils.data import DataLoader

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
            labels.append(speaker_id)

        return torch.cat(waveforms), torch.tensor(labels)

    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)

def evaluate_model():
    # Load dataset
    data, labels = load_dataset()

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Load the trained model
    model = torch.load('model.pth')
    model.eval()

    # Perform evaluation on the test set
    with torch.no_grad():
        predictions = model(torch.tensor(X_test, dtype=torch.float32))
        predicted_labels = torch.argmax(predictions, dim=1).numpy()

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
