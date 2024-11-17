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


data_dir = './data/arc_data/ARC-AGI-master/data/'
# training_dir = os.path.join(data_dir, 'training')
evaluation_dir = os.path.join(data_dir, 'evaluation')

val_split = 0.2
random_seed = 42
random.seed(random_seed)

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# train_files = [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.endswith('.json')]
evaluation_files = [os.path.join(evaluation_dir, f) for f in os.listdir(evaluation_dir) if f.endswith('.json')]

# random.shuffle(train_files)
# train_files, val_files = train_test_split(train_files, test_size=val_split, random_state=random_seed)

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

# train_dataset = process_files(train_files, key='train')
# val_dataset = process_files(val_files, key='train')
eval_dataset = process_files(evaluation_files, key='test')

output_dir = './data/arc_data/processed_data'
os.makedirs(output_dir, exist_ok=True)

# with open(os.path.join(output_dir, 'train_dataset.json'), 'w') as f:
#     json.dump(train_dataset, f, indent=4)

# with open(os.path.join(output_dir, 'val_dataset.json'), 'w') as f:
#     json.dump(val_dataset, f, indent=4)

with open(os.path.join(output_dir, 'eval_dataset.json'), 'w') as f:
    json.dump(eval_dataset, f, indent=4)

# print(f"Training samples: {len(train_dataset)}")
# print(f"Validation samples: {len(val_dataset)}")
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

# with open(os.path.join(output_dir, 'train_dataset.json'), 'r') as f:
#     train_dataset = json.load(f)

# with open(os.path.join(output_dir, 'val_dataset.json'), 'r') as f:
#     val_dataset = json.load(f)

with open(os.path.join(output_dir, 'eval_dataset.json'), 'r') as f:
    eval_dataset = json.load(f)

# padded_train_dataset = pad_dataset_uniform(train_dataset, target_size=30, pad_value=10)
# padded_val_dataset = pad_dataset_uniform(val_dataset, target_size=30, pad_value=10)
padded_eval_dataset = pad_dataset_uniform(eval_dataset, target_size=30, pad_value=10)

# with open(os.path.join(output_dir, 'padded_train_dataset.json'), 'w') as f:
#     json.dump(padded_train_dataset, f, indent=4)

# with open(os.path.join(output_dir, 'padded_val_dataset.json'), 'w') as f:
#     json.dump(padded_val_dataset, f, indent=4)

with open(os.path.join(output_dir, 'padded_eval_dataset.json'), 'w') as f:
    json.dump(padded_eval_dataset, f, indent=4)

print("Padding complete!")
# print(f"Padded Training samples: {len(padded_train_dataset)}")
# print(f"Padded Validation samples: {len(padded_val_dataset)}")
print(f"Padded Evaluation samples: {len(padded_eval_dataset)}")


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

# flattened_train_dataset = flatten_dataset(padded_train_dataset)
# flattened_val_dataset = flatten_dataset(padded_val_dataset)
flattened_eval_dataset = flatten_dataset(padded_eval_dataset)

output_dir = './data/arc_data/processed_data'

# with open(os.path.join(output_dir, 'flattened_train_dataset.json'), 'w') as f:
#     json.dump(flattened_train_dataset, f, indent=4)

# with open(os.path.join(output_dir, 'flattened_val_dataset.json'), 'w') as f:
#     json.dump(flattened_val_dataset, f, indent=4)

with open(os.path.join(output_dir, 'flattened_eval_dataset.json'), 'w') as f:
    json.dump(flattened_eval_dataset, f, indent=4)

print("Flattened datasets saved!")
# print(f"Flattened Training samples: {len(flattened_train_dataset)}")
# print(f"Flattened Validation samples: {len(flattened_val_dataset)}")
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

eval_inputs = FlattenedARCDataset(flattened_eval_dataset, is_input=True)
eval_outputs = FlattenedARCDataset(flattened_eval_dataset, is_input=False)

eval_input_loader = DataLoader(eval_inputs, batch_size=BATCH_SIZE, shuffle=False)
eval_output_loader = DataLoader(eval_outputs, batch_size=BATCH_SIZE, shuffle=False)


# In[ ]:


def strict_match_evaluation(model, input_loader, output_loader, device):
    model.eval()
    strict_matches = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in tqdm(zip(input_loader, output_loader), desc="Strict Match Evaluation"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            predicted_outputs = torch.round(outputs).cpu().numpy()
            target_outputs = targets.cpu().numpy()

            for predicted, target in zip(predicted_outputs, target_outputs):
                if np.array_equal(predicted, target):
                    strict_matches += 1
                total_samples += 1

    accuracy = (strict_matches / total_samples) * 100
    print(f"Strict Match Accuracy: {accuracy:.2f}% ({strict_matches}/{total_samples})")
    return accuracy

model_path = os.path.join("./saved_models", "best_meta_model.pth")
model.load_state_dict(torch.load(model_path))
# model.load_state_dict(torch.load("best_meta_model.pth"))
model.to(DEVICE)

strict_match_accuracy = strict_match_evaluation(model, eval_input_loader, eval_output_loader, DEVICE)

