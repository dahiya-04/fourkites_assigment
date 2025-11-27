import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
import numpy as np
import os
import json
from landscape_prober import LossLandscapeProber

# --- Configuration ---
NUM_EPOCHS = 5
BATCH_SIZE = 64
DATA_DIR = './data'
MODELS_DIR = './trained_models'
RESULTS_FILE = './experiment_results.json'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definition (Simple CNN for MNIST) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 320)
        x = self.fc(x)
        return x

# --- Data Loading ---
def load_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load the training data
    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    
    # Use a subset of the training data for the full loss calculation in the prober
    # This is to keep the full loss calculation feasible for the HVP.
    train_subset_size = 1000
    train_subset, _ = random_split(train_dataset, [train_subset_size, len(train_dataset) - train_subset_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    prober_loader = DataLoader(train_subset, batch_size=len(train_subset), shuffle=False) # Single batch for full loss
    
    return train_loader, test_loader, prober_loader

# --- Training Function ---
def train_model(model, optimizer, criterion, train_loader, epochs):
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

# --- Evaluation Function ---
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

# --- Main Experiment Logic ---
def run_experiment():
    print(f"Running experiment on device: {DEVICE}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    train_loader, test_loader, prober_loader = load_mnist_data()
    criterion = nn.CrossEntropyLoss()
    
    experiment_configs = [
        # Config 1: Sharp Minimum (High LR, Low Weight Decay)
        {"name": "Sharp_SGD_HighLR", "optimizer": optim.SGD, "lr": 0.1, "weight_decay": 1e-5},
        # Config 2: Flat Minimum (Low LR, High Weight Decay)
        {"name": "Flat_SGD_LowLR", "optimizer": optim.SGD, "lr": 0.001, "weight_decay": 1e-3},
        # Config 3: Adam (Often finds flatter minima than SGD with same LR)
        {"name": "Adam_Default", "optimizer": optim.Adam, "lr": 0.001, "weight_decay": 1e-5},
    ]
    
    results = []
    
    for config in experiment_configs:
        print(f"\n--- Running Configuration: {config['name']} ---")
        
        # 1. Initialize and Train Model
        model = SimpleCNN().to(DEVICE)
        optimizer = config["optimizer"](model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        
        train_model(model, optimizer, criterion, train_loader, NUM_EPOCHS)
        
        # 2. Evaluate Generalization
        test_loss, accuracy = evaluate_model(model, test_loader, criterion)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
        
        # 3. Probing Loss Landscape (Sharpness)
        prober = LossLandscapeProber(model, criterion, prober_loader)
        
        # Estimate max eigenvalue (Sharpness)
        try:
            lambda_max, max_eigenvector = prober.estimate_max_eigenvalue(num_iterations=20)
            print(f"Maximum Eigenvalue (Sharpness): {lambda_max:.4f}")
        except Exception as e:
            print(f"Error during sharpness estimation: {e}")
            lambda_max = None
            max_eigenvector = None
            
        # 4. Record Results
        result = {
            "name": config["name"],
            "optimizer": config["optimizer"].__name__,
            "lr": config["lr"],
            "weight_decay": config["weight_decay"],
            "test_loss": test_loss,
            "test_accuracy": accuracy,
            "sharpness_lambda_max": lambda_max,
        }
        results.append(result)
        
        # 5. Save Model and Max Eigenvector for Visualization (Phase 5)
        model_path = os.path.join(MODELS_DIR, f"{config['name']}_model.pth")
        torch.save(model.state_dict(), model_path)
        
        if max_eigenvector is not None:
            eigenvector_path = os.path.join(MODELS_DIR, f"{config['name']}_eigenvector.pt")
            torch.save(max_eigenvector, eigenvector_path)
            result["eigenvector_path"] = eigenvector_path
            
    # 6. Save all results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nExperiment complete. Results saved to {RESULTS_FILE}")

if __name__ == '__main__':
    # Create the necessary files for the experiment to run
    # The SimpleCNN class is needed in the same scope as the main script
    # to be loaded correctly by torch.load in a real scenario, but here 
    # we just need to ensure the script runs.
    
    # We need to ensure the prober is available. It was written in a separate file.
    # The execution environment should handle the import, but for safety, 
    # we'll assume the environment is set up correctly.
    
    # Since the execution of this script will take time and involves training, 
    # I will execute it now.
    run_experiment()
