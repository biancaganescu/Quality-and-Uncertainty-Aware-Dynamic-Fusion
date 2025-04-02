import sys 
import os 
sys.path.append(os.getcwd())
sys.path.append('/home/bianca/Code/MultiBench/mm_health_bench/mmhb')
from torch.utils.data import random_split, Subset
import torch
from torch.utils.data import DataLoader, Dataset
from mm_health_bench.mmhb.loader import ChestXDataset
from mm_health_bench.mmhb.utils import Config
from torch.utils.data import WeightedRandomSampler



def chestx_collate_fn(batch):

    images = [sample[0][0] for sample in batch]
    reports = [sample[0][1] for sample in batch]
    targets = [sample[1] for sample in batch]

    
    images = torch.stack(images, dim=0)
    reports = torch.stack(reports, dim=0)#(batch, s)
    targets = torch.tensor(targets, dtype=torch.float) #(batch, num_labels)
    return reports, images, targets



def get_data(batch_size=32, num_workers=4, indices_path="split_indices.pth"):
    config = Config("./mm_health_bench/config/config.yml").read()
    chestx_dataset = ChestXDataset(data_path="./mm_health_bench/data/chestx", max_seq_length=256)
    
    total_length = len(chestx_dataset)
    train_length = int(0.8 * total_length)
    val_length = int(0.1 * total_length)
    test_length = total_length - train_length - val_length

    if os.path.exists(indices_path):
        indices = torch.load(indices_path)
        train_indices = indices['train']
        val_indices = indices['val']
        test_indices = indices['test']
        
        train_dataset = Subset(chestx_dataset, train_indices)
        val_dataset = Subset(chestx_dataset, val_indices)
        test_dataset = Subset(chestx_dataset, test_indices)
        print(f"Loaded split indices from {indices_path}")
    else:
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            chestx_dataset, [train_length, val_length, test_length]
        )
        indices = {
            'train': train_dataset.indices,
            'val': val_dataset.indices,
            'test': test_dataset.indices,
        }
        torch.save(indices, indices_path)
        print(f"Saved split indices to {indices_path}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=chestx_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=chestx_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=chestx_collate_fn
    )

    return train_loader, val_loader, test_loader



def analyze_class_distribution(data_loader):
    class_counts = torch.zeros(14)
    total_samples = 0
    
    for batch in data_loader:
        _, _, targets = batch
        class_counts += targets.sum(dim=0)
        total_samples += targets.size(0)
    
    class_percentages = (class_counts / total_samples) * 100
    
    print("Class distribution:")
    for i, percentage in enumerate(class_percentages):
        print(f"Class {i}: {percentage:.2f}% positive")
    
    print(f"Most common class: {torch.argmax(class_counts)} ({torch.max(class_counts)}/{total_samples} samples)")
    print(f"Least common class: {torch.argmin(class_counts)} ({torch.min(class_counts)}/{total_samples} samples)")

def get_data_balanced(batch_size=64, num_workers=4):
    config = Config("./mm_health_bench/config/config.yml").read()
    chestx_dataset = ChestXDataset(data_path="./mm_health_bench/data/chestx", max_seq_length=256)
    print(chestx_dataset[0][1])
    
    total_length = len(chestx_dataset)
    train_length = int(0.8 * total_length)
    val_length = int(0.1 * total_length)
    test_length = total_length - train_length - val_length
    
    train_dataset, val_dataset, test_dataset = random_split(chestx_dataset, [train_length, val_length, test_length])
    
    train_labels = []
    for i in range(len(train_dataset)):
        sample = train_dataset[i]

        if isinstance(sample, tuple) and len(sample) == 2:
            _, label = sample
        elif isinstance(sample, tuple) and len(sample) == 3:
            _, _, label = sample
        else:
            print(f"Unexpected sample structure: {type(sample)}")
            continue

        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.float)
        
        train_labels.append(label)

    train_labels = torch.stack(train_labels)
    
    weights = torch.ones(len(train_dataset))
    
    rare_classes = [2, 3, 5, 10, 11, 12]  
    
    for i, label in enumerate(train_labels):
        for cls in rare_classes:
            if label[cls] > 0:
                weights[i] *= 10 
        
        very_rare = [3, 11, 12]
        for cls in very_rare:
            if label[cls] > 0:
                weights[i] *= 2  
    
    train_sampler = WeightedRandomSampler(
        weights, 
        num_samples=len(train_dataset) * 3, 
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=num_workers, 
        collate_fn=chestx_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=chestx_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=chestx_collate_fn
    )
    
    return train_loader, val_loader, test_loader

def compute_pos_weights(train_loader):
    
    num_pos = torch.zeros(14)  
    num_neg = torch.zeros(14)
    
    for _, _, targets in train_loader:
        num_pos += targets.sum(dim=0)
        num_neg += (1 - targets).sum(dim=0)
    
    pos_weights = num_neg / (num_pos + 1e-5)  
    
    pos_weights = torch.clamp(pos_weights, min=0.5, max=10.0)
    
    return pos_weights.cuda()

if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_data()

    for batch in train_loader:
    # If your loader returns just data
        print(batch[1].shape)
        break
    # print("TRAINING SET:")
    # analyze_class_distribution(train_loader)
    # print("\nVALIDATION SET:")
    # analyze_class_distribution(val_loader)
    # print("\nTEST SET:")
    # analyze_class_distribution(test_loader)

