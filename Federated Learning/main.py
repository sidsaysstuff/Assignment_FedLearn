import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import time
import pickle
import matplotlib.pyplot as plt

# ---- CUDA SETUP ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---- Parameters ----
NUM_CLIENTS = 1000
CLIENTS_PER_ROUND = 100
BATCH_SIZE = 32
EPOCHS_PER_CLIENT = 10
TARGET_ACCURACY = 0.75
MAX_ROUNDS = 150
DEVICE_FAILURE_PROB = 0.1  # 10% chance a selected device drops out per round

CLOUD_HOURLY_RATE = 0.526  # USD/hour
DATA_SIZE_GB = 0.636
STORAGE_RATE = 0.023       # USD/GB/month

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ---- Data Preparation ----
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

data_dir = '../animals10'
full_dataset = datasets.ImageFolder(data_dir, transform=transform)
num_classes = len(full_dataset.classes)

# Split into train/test (80/20)
num_total = len(full_dataset)
indices = list(range(num_total))
random.shuffle(indices)
split = int(0.8 * num_total)
train_indices, test_indices = indices[:split], indices[split:]
train_dataset = Subset(full_dataset, train_indices)
test_dataset = Subset(full_dataset, test_indices)

# ---- Server's Skeleton Data ----
SKELETON_SIZE = int(0.02 * len(train_dataset))
server_indices = train_indices[:SKELETON_SIZE]
server_dataset = Subset(full_dataset, server_indices)
client_indices_pool = train_indices[SKELETON_SIZE:]

# ---- Simulate Clients ----
def split_client_indices(indices, num_clients):
    random.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return [list(split) for split in splits]

client_indices_list = split_client_indices(client_indices_pool, NUM_CLIENTS)
client_datasets = [Subset(full_dataset, idxs) for idxs in client_indices_list]

def get_new_data_indices(all_indices, used_indices, per_client):
    available = list(set(all_indices) - set(used_indices))
    random.shuffle(available)
    splits = np.array_split(available, NUM_CLIENTS)
    return [list(split[:per_client]) for split in splits]

battery_levels = np.random.rand(NUM_CLIENTS)
comm_types = np.random.choice(['free', 'paid'], NUM_CLIENTS)
compute_caps = np.random.rand(NUM_CLIENTS)

def select_clients(battery_levels, comm_types, compute_caps, k):
    eligible = []
    for i in range(len(battery_levels)):
        if battery_levels[i] > 0.5 and comm_types[i] == 'free' and compute_caps[i] > 0.5:
            eligible.append(i)
    if len(eligible) >= k:
        return random.sample(eligible, k)
    selected = eligible
    pool = [i for i in range(len(battery_levels)) if i not in selected]
    selected += random.sample(pool, k - len(selected))
    return selected

# ---- Model Definition ----
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_model():
    model = SimpleCNN(num_classes)
    return model.to(device)

# ---- Persistent Storage ----
def save_model_checkpoint(model, filename='global_model_ckpt.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(model.state_dict(), f)

def load_model_checkpoint(model, filename='global_model_ckpt.pkl'):
    with open(filename, 'rb') as f:
        model.load_state_dict(pickle.load(f))
    return model

# ---- Federated Learning Functions ----
def train_local(model, dataloader, epochs=1, lr=0.01):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    # Move state_dict to CPU for communication
    return {k: v.cpu() for k, v in model.state_dict().items()}

def aggregate_models(global_model, client_states):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        tensors = [client_states[i][key].float() for i in range(len(client_states))]
        global_dict[key] = torch.stack(tensors, 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

def model_size_bytes(model):
    return sum(p.numel() for p in model.parameters()) * 4  # float32

# ---- Evaluation ----
def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total

# ---- Federated Learning Main Loop with Metrics Collection ----
global_model = get_model()

# Server trains skeleton model (persistent storage)
server_loader = DataLoader(server_dataset, batch_size=32, shuffle=True)
global_model.train()
optimizer = optim.SGD(global_model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
for epoch in range(2):
    for data, target in server_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = global_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
save_model_checkpoint(global_model)

used_indices_per_client = [set(idxs) for idxs in client_indices_list]
all_train_indices = set(client_indices_pool)

# Metrics for plotting
round_accuracies = []
round_comm_costs = []
round_client_counts = []

fed_comm_cost = 0
start_time = time.time()
round_num = 0

while round_num < MAX_ROUNDS:
    round_num += 1
    # Selection
    selected = select_clients(battery_levels, comm_types, compute_caps, CLIENTS_PER_ROUND)

    global_model = get_model()
    global_model = load_model_checkpoint(global_model)

    client_states = []
    new_data_per_client = get_new_data_indices(
        list(all_train_indices),
        set().union(*used_indices_per_client),
        per_client=2
    )
    clients_updated_this_round = 0
    for idx, client_id in enumerate(selected):
        if random.random() < DEVICE_FAILURE_PROB:
            continue
        if new_data_per_client[client_id]:
            used_indices_per_client[client_id].update(new_data_per_client[client_id])
            client_datasets[client_id] = Subset(full_dataset, list(used_indices_per_client[client_id]))
        local_model = get_model()
        local_model.load_state_dict(global_model.state_dict())
        loader = DataLoader(client_datasets[client_id], batch_size=BATCH_SIZE, shuffle=True)
        client_state = train_local(local_model, loader, epochs=EPOCHS_PER_CLIENT)
        client_states.append(client_state)
        fed_comm_cost += model_size_bytes(local_model)
        clients_updated_this_round += 1
    if not client_states:
        round_accuracies.append(round_accuracies[-1] if round_accuracies else 0)
        round_comm_costs.append(fed_comm_cost)
        round_client_counts.append(0)
        continue
    global_model = aggregate_models(global_model, client_states)
    save_model_checkpoint(global_model)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    fed_acc = evaluate(global_model, test_loader)
    print(f"Round {round_num} | Test Accuracy: {fed_acc:.4f}")
    round_accuracies.append(fed_acc)
    round_comm_costs.append(fed_comm_cost)
    round_client_counts.append(clients_updated_this_round)
    if fed_acc >= TARGET_ACCURACY:
        print(f"Target accuracy {TARGET_ACCURACY} reached at round {round_num}. Stopping early.")
        break

end_time = time.time()
training_time_seconds = end_time - start_time

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
fed_acc = evaluate(global_model, test_loader)
print(f"\nFinal Federated Accuracy: {fed_acc:.4f}")
print(f"Federated Communication Cost (bytes): {fed_comm_cost}")

training_time_hours = training_time_seconds / 3600
compute_cost = CLOUD_HOURLY_RATE * training_time_hours
storage_cost = STORAGE_RATE * DATA_SIZE_GB
total_cost = compute_cost + storage_cost

print(f"\n--- Federated ML Cloud Simulation Cost ---")
print(f"Total training time: {training_time_hours:.2f} hours")
print(f"Compute cost (@${CLOUD_HOURLY_RATE}/hr): ${compute_cost:.2f}")
print(f"Storage cost ({DATA_SIZE_GB} GB @${STORAGE_RATE}/GB/month): ${storage_cost:.2f}")
print(f"Total simulation cost: ${total_cost:.2f}")

# ---- Save Metrics for Later Analysis ----
metrics = {
    'round_accuracies': round_accuracies,
    'round_comm_costs': round_comm_costs,
    'round_client_counts': round_client_counts,
    'training_time_seconds': training_time_seconds
}
with open('global_model_ckpt.pkl', 'wb') as f:
    pickle.dump(metrics, f)
print("Metrics saved to global_model_ckpt.pkl")

# ---- Plotting Results ----
rounds = list(range(1, len(round_accuracies) + 1))

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(rounds, round_accuracies, marker='o')
plt.xlabel('FL Round')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy vs FL Round')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(rounds, round_comm_costs, marker='o', color='orange')
plt.xlabel('FL Round')
plt.ylabel('Cumulative Communication Cost (bytes)')
plt.title('Communication Cost vs FL Round')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(rounds, round_client_counts, marker='o', color='green')
plt.xlabel('FL Round')
plt.ylabel('Client Updates')
plt.title('Clients Updated per Round')
plt.grid(True)

plt.tight_layout()
plt.show()
