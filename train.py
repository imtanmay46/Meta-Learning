#!/usr/bin/env python
# coding: utf-8

# Meta Learning
# 
# ARC
# 
# Tanmay Singh
# 2021569
# CSAI
# Class of '25

# In[ ]:


import os
import json
import random
import torch
import numpy as np
import torch.nn as nn
import mplcyberpunk as mcy
from tqdm.auto import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap, Normalize


# In[ ]:


def load_task(task_path):
    with open(task_path, 'r') as f:
        task = json.load(f)
    return task

def visualize_task(task, title_prefix="Original", num_samples=None):
    if num_samples is not None:
        task = task[:num_samples]
    
    for idx, pair in enumerate(task):
        input_grid = np.array(pair["input"])
        output_grid = np.array(pair["output"])
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        cmap = ListedColormap([
            '#000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
            '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
        ])
        norm = Normalize(vmin=0, vmax=9)

        axes[0].imshow(input_grid, cmap=cmap, norm=norm)
        axes[0].set_title(f"{title_prefix} Input {idx+1}")
        axes[1].imshow(output_grid, cmap=cmap, norm=norm)
        axes[1].set_title(f"{title_prefix} Output {idx+1}")

        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

def plot_task(task):
    examples = task['train']
    n_examples = len(examples)
    cmap = ListedColormap([
        '#000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ])
    norm = Normalize(vmin=0, vmax=9)
    fig, axes = plt.subplots(2, n_examples, figsize=(n_examples * 4, 8))
    for i, example in enumerate(examples):
        axes[0, i].imshow(example['input'], cmap=cmap, norm=norm)
        axes[1, i].imshow(example['output'], cmap=cmap, norm=norm)
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    plt.show()


# In[ ]:


data_dir = './data/arc_data/ARC-AGI-master/data/'
training_dir = os.path.join(data_dir, 'training')
evaluation_dir = os.path.join(data_dir, 'evaluation')

val_split = 0.2
random_seed = 42
random.seed(random_seed)

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

train_files = [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.endswith('.json')]
evaluation_files = [os.path.join(evaluation_dir, f) for f in os.listdir(evaluation_dir) if f.endswith('.json')]

random.shuffle(train_files)
train_files, val_files = train_test_split(train_files, test_size=val_split, random_state=random_seed)

def process_files(file_list, key='train'):
    dataset = []
    for file_path in file_list:
        data = load_json(file_path)
        for item in data[key]:
            dataset.append({
                'input': item['input'],
                'output': item['output']
            })
    return dataset

train_dataset = process_files(train_files, key='train')
val_dataset = process_files(val_files, key='train')
eval_dataset = process_files(evaluation_files, key='test')

output_dir = './data/arc_data/processed_data'
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, 'train_dataset.json'), 'w') as f:
    json.dump(train_dataset, f, indent=4)

with open(os.path.join(output_dir, 'val_dataset.json'), 'w') as f:
    json.dump(val_dataset, f, indent=4)

with open(os.path.join(output_dir, 'eval_dataset.json'), 'w') as f:
    json.dump(eval_dataset, f, indent=4)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Evaluation samples: {len(eval_dataset)}")


# In[ ]:


def pad_grid_uniform(grid, target_size=30, pad_value=10):
    rows, cols = len(grid), len(grid[0]) if grid else 0

    row_padding = (target_size - rows) // 2
    col_padding = (target_size - cols) // 2

    row_padding_extra = (target_size - rows) % 2
    col_padding_extra = (target_size - cols) % 2

    padded_grid = np.full((target_size, target_size), pad_value, dtype=int)

    padded_grid[
        row_padding : row_padding + rows, 
        col_padding : col_padding + cols
    ] = grid

    return padded_grid.tolist()

def pad_dataset_uniform(dataset, target_size=30, pad_value=10):
    padded_dataset = []
    for task in dataset:
        padded_task = {
            "input": pad_grid_uniform(task["input"], target_size, pad_value),
            "output": pad_grid_uniform(task["output"], target_size, pad_value)
        }
        padded_dataset.append(padded_task)
    return padded_dataset

output_dir = './data/arc_data/processed_data'

with open(os.path.join(output_dir, 'train_dataset.json'), 'r') as f:
    train_dataset = json.load(f)

with open(os.path.join(output_dir, 'val_dataset.json'), 'r') as f:
    val_dataset = json.load(f)

with open(os.path.join(output_dir, 'eval_dataset.json'), 'r') as f:
    eval_dataset = json.load(f)

padded_train_dataset = pad_dataset_uniform(train_dataset, target_size=30, pad_value=10)
padded_val_dataset = pad_dataset_uniform(val_dataset, target_size=30, pad_value=10)
padded_eval_dataset = pad_dataset_uniform(eval_dataset, target_size=30, pad_value=10)

with open(os.path.join(output_dir, 'padded_train_dataset.json'), 'w') as f:
    json.dump(padded_train_dataset, f, indent=4)

with open(os.path.join(output_dir, 'padded_val_dataset.json'), 'w') as f:
    json.dump(padded_val_dataset, f, indent=4)

with open(os.path.join(output_dir, 'padded_eval_dataset.json'), 'w') as f:
    json.dump(padded_eval_dataset, f, indent=4)

print("Padding complete!")
print(f"Padded Training samples: {len(padded_train_dataset)}")
print(f"Padded Validation samples: {len(padded_val_dataset)}")
print(f"Padded Evaluation samples: {len(padded_eval_dataset)}")


# In[ ]:


task_path = './data/arc_data/processed_data/train_dataset.json'
task = load_task(task_path)

visualize_task(task, title_prefix="Original", num_samples = 10)


# In[ ]:


padded_task_path = './data/arc_data/processed_data/padded_train_dataset.json'
padded_task = load_task(padded_task_path)

visualize_task(padded_task, title_prefix="Padded", num_samples = 10)


# In[ ]:


def flatten_grid(grid):
    return [cell for row in grid for cell in row]

def flatten_dataset(dataset):
    flattened_dataset = []
    for task in dataset:
        flattened_task = {
            "input": flatten_grid(task["input"]),
            "output": flatten_grid(task["output"])
        }
        flattened_dataset.append(flattened_task)
    return flattened_dataset

flattened_train_dataset = flatten_dataset(padded_train_dataset)
flattened_val_dataset = flatten_dataset(padded_val_dataset)
flattened_eval_dataset = flatten_dataset(padded_eval_dataset)

output_dir = './data/arc_data/processed_data'

with open(os.path.join(output_dir, 'flattened_train_dataset.json'), 'w') as f:
    json.dump(flattened_train_dataset, f, indent=4)

with open(os.path.join(output_dir, 'flattened_val_dataset.json'), 'w') as f:
    json.dump(flattened_val_dataset, f, indent=4)

with open(os.path.join(output_dir, 'flattened_eval_dataset.json'), 'w') as f:
    json.dump(flattened_eval_dataset, f, indent=4)

print("Flattened datasets saved!")
print(f"Flattened Training samples: {len(flattened_train_dataset)}")
print(f"Flattened Validation samples: {len(flattened_val_dataset)}")
print(f"Flattened Evaluation samples: {len(flattened_eval_dataset)}")


# In[ ]:


BATCH_SIZE = 4
NUM_EPOCHS = 100
INNER_LR = 1e-4
OUTER_LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


class FlattenedARCDataset(Dataset):
    def __init__(self, data, is_input=True):
        self.data = data
        self.is_input = is_input

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_input:
            grid = self.data[idx]["input"]
        else:
            grid = self.data[idx]["output"]
        return torch.tensor(grid, dtype=torch.float32)

train_inputs = FlattenedARCDataset(flattened_train_dataset, is_input=True)
train_outputs = FlattenedARCDataset(flattened_train_dataset, is_input=False)

val_inputs = FlattenedARCDataset(flattened_val_dataset, is_input=True)
val_outputs = FlattenedARCDataset(flattened_val_dataset, is_input=False)

eval_inputs = FlattenedARCDataset(flattened_eval_dataset, is_input=True)
eval_outputs = FlattenedARCDataset(flattened_eval_dataset, is_input=False)

train_input_loader = DataLoader(train_inputs, batch_size=BATCH_SIZE, shuffle=True)
train_output_loader = DataLoader(train_outputs, batch_size=BATCH_SIZE, shuffle=True)

val_input_loader = DataLoader(val_inputs, batch_size=BATCH_SIZE, shuffle=False)
val_output_loader = DataLoader(val_outputs, batch_size=BATCH_SIZE, shuffle=False)

eval_input_loader = DataLoader(eval_inputs, batch_size=BATCH_SIZE, shuffle=False)
eval_output_loader = DataLoader(eval_outputs, batch_size=BATCH_SIZE, shuffle=False)


# In[ ]:


class MAMLTrainingPipeline:
    def __init__(self, model, train_input_loader, train_output_loader, val_input_loader, val_output_loader, outer_optimizer, device, inner_lr=INNER_LR):
        self.model = model
        self.train_input_loader = train_input_loader
        self.train_output_loader = train_output_loader
        self.val_input_loader = val_input_loader
        self.val_output_loader = val_output_loader
        self.outer_optimizer = outer_optimizer
        self.device = device
        self.inner_lr = inner_lr

    def compute_elementwise_loss(self, predicted, actual):
        differences = torch.abs(predicted - actual)
        total_loss = differences.sum()
        avg_loss = total_loss / predicted.numel()
        return avg_loss

    def inner_loop(self, inputs, targets):
        task_model = EncoderDecoderModel().to(self.device)
        task_model.load_state_dict(self.model.state_dict())
        inner_optimizer = optim.Adam(task_model.parameters(), lr=self.inner_lr)

        task_model.train()
        for _ in range(5):
            inner_optimizer.zero_grad()
            outputs = task_model(inputs)
            loss = self.compute_elementwise_loss(outputs, targets)
            loss.backward()
            inner_optimizer.step()

        return task_model

    def outer_loop(self, num_epochs):
        train_losses = []

        for epoch in tqdm(range(1, num_epochs + 1), desc="Meta-Training"):
            self.model.train()
            total_meta_loss = 0.0

            for inputs, targets in tqdm(zip(self.train_input_loader, self.train_output_loader), desc=f"Epoch {epoch} - Inner Loop", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                adapted_model = self.inner_loop(inputs, targets)

                adapted_model.eval()
                meta_outputs = adapted_model(inputs)
                meta_loss = self.compute_elementwise_loss(meta_outputs, targets)

                meta_loss.backward()
                total_meta_loss += meta_loss.item()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()

            train_losses.append(total_meta_loss / len(self.train_input_loader))
            print(f"Epoch {epoch}: Train Loss = {train_losses[-1]:.4f}")

            epoch_model_path = os.path.join("./saved_models", f"meta_model_epoch_{epoch}.pth")
            torch.save(self.model.state_dict(), epoch_model_path)
            # torch.save(self.model.state_dict(), f"meta_model_epoch_{epoch}.pth")

        print("Meta-Training completed. Computing final validation loss.")
        val_losses = self.validate()
        print(f"Final Validation Loss: {sum(val_losses) / len(val_losses):.4f}")

        best_model_path = os.path.join("./saved_models", "best_meta_model.pth")
        torch.save(self.model.state_dict(), best_model_path)
        # torch.save(self.model.state_dict(), "best_meta_model.pth")
        print("Model saved as 'best_meta_model.pth'.")

        return train_losses, val_losses

    def validate(self):
        self.model.eval()
        batch_losses = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(zip(self.val_input_loader, self.val_output_loader), 1):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.compute_elementwise_loss(outputs, targets)
                batch_losses.append(loss.item())
                print(f"Validation Batch {batch_idx}: Loss = {loss.item():.4f}")

        return batch_losses


# In[ ]:


model = EncoderDecoderModel().to(DEVICE)
outer_optimizer = optim.AdamW(model.parameters(), lr=OUTER_LR, weight_decay=0.01)

pipeline = MAMLTrainingPipeline(
    model,
    train_input_loader,
    train_output_loader,
    val_input_loader,
    val_output_loader,
    outer_optimizer,
    DEVICE
)

train_losses, val_losses = pipeline.outer_loop(num_epochs=NUM_EPOCHS)


# In[ ]:


plt.style.use("cyberpunk")
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Meta-Train Loss", color="gold")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Meta-Training Loss")
plt.legend()
mcy.add_gradient_fill()
plt.show()

plt.style.use("cyberpunk")
plt.figure(figsize=(10, 5))
plt.plot(val_losses, label="Validation Loss (Per Batch)", color="coral")
plt.xlabel("Batches")
plt.ylabel("Loss")
plt.title("Validation Loss Per Batch")
plt.legend()
mcy.add_gradient_fill()
plt.show()

print(f"Meta-Training completed. Final Validation Loss: {sum(val_losses) / len(val_losses):.4f}")

