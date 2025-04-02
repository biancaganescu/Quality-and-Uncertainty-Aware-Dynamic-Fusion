import torch
import torch.nn.functional as F
import numpy as np
import random
import sys 
import os 
sys.path.append(os.getcwd())
sys.path.append('/home/bianca/Code/MultiBench/mm_health_bench/mmhb')
from torch.utils.data import Subset
import torch
from torch.utils.data import DataLoader
import hashlib
import json
import torchvision

class NoiseAugmenter:


    @staticmethod
    def add_gaussian_noise_to_image(image, mean=0.0, std=0.1):
        noise = torch.randn_like(image) * std + mean
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0, 1)
    
    @staticmethod
    def add_salt_and_pepper_noise(image, amount=0.05):
        noisy_image = image.clone()
        
        salt_mask = torch.rand_like(image) < (amount/2)
        noisy_image[salt_mask] = 1.0
        
        pepper_mask = torch.rand_like(image) < (amount/2)
        noisy_image[pepper_mask] = 0.0
        
        return noisy_image
    
    
    @staticmethod
    def reduce_image_quality(image, blur_factor=1.5, noise_level=0):
        kernel_size = int(blur_factor * 2) * 2 + 1  # Ensure odd kernel size
        blurred = torchvision.transforms.functional.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], 
                                 sigma=[blur_factor, blur_factor])
        
        noisy_blurred = NoiseAugmenter.add_gaussian_noise_to_image(blurred, std=noise_level)
        
        return noisy_blurred
    
    @staticmethod
    def mask_image_regions(image, mask_size=0.2, num_masks=3):
        masked_image = image.clone()
        batch_size, channels, height, width = image.shape
        
        mask_h = int(height * mask_size)
        mask_w = int(width * mask_size)
        
        for b in range(batch_size):
            for _ in range(num_masks):
                top = random.randint(0, height - mask_h)
                left = random.randint(0, width - mask_w)
                
                mean_value = image[b].mean()
                masked_image[b, :, top:top+mask_h, left:left+mask_w] = mean_value
        
        return masked_image
    
    
    
    @staticmethod
    def add_word_dropout(text, dropout_prob=0.3):
        
        if len(text.shape) == 3: 
            mask = torch.rand(text.shape[0], text.shape[1], 1, device=text.device) >= dropout_prob
            return text * mask
        else:
            non_padding_mask = (text != 0)
            dropout_mask = (torch.rand_like(text.float()) >= dropout_prob)
            combined_mask = non_padding_mask & dropout_mask

            result = text * combined_mask.long()
            return result
    
    @staticmethod
    def swap_words(text, swap_prob=0.3):
        result = text.clone()
        batch_size, seq_len = text.shape
        
        for b in range(batch_size):
            for i in range(seq_len - 1):
                if text[b, i] == 0 or text[b, i+1] == 0:
                    continue
                    
                if random.random() < swap_prob:
                    result[b, i], result[b, i+1] = text[b, i+1], text[b, i]
        
        return result
    
    @staticmethod
    def corrupt_text(text, corruption_prob=0.5, pad_token_id=0):

        corrupted = text.clone()
        batch_size, seq_len = text.shape
        
        vocab_size = torch.max(text) + 1
        
        for b in range(batch_size):
            for i in range(seq_len):
                if text[b, i] == pad_token_id:
                    continue
                    
                if random.random() < corruption_prob:
                    random_token = random.randint(1, vocab_size - 1)
                    corrupted[b, i] = random_token
        
        return corrupted



def get_config_hash(config):
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


def ensure_indices_dir():
    indices_dir = os.path.join(os.getcwd(), "noise_indices")
    if not os.path.exists(indices_dir):
        os.makedirs(indices_dir)
    return indices_dir


def get_noise_indices(dataset_size, batch_size, config, split_name, seed=42):
    random.seed(seed)
    
    indices_dir = ensure_indices_dir()
    
    config_hash = get_config_hash(config)
    indices_filename = f"{split_name}_{config_hash}_{dataset_size}_{batch_size}.json"
    indices_path = os.path.join(indices_dir, indices_filename)
    
    if os.path.exists(indices_path):
        with open(indices_path, 'r') as f:
            batch_noise_indices = json.load(f)
        return batch_noise_indices

    image_noise_percentage = config.get("image_noise_percentage", 0)
    text_noise_percentage = config.get("text_noise_percentage", 0)
    

    num_batches = (dataset_size + batch_size - 1) // batch_size 
    batch_noise_indices = []
    
    for i in range(num_batches):
        current_batch_size = min(batch_size, dataset_size - i * batch_size)
        batch_indices = list(range(current_batch_size))
        
        batch_dict = {
            "image_indices": [],
            "text_indices": []
        }
        
        if image_noise_percentage > 0:
            num_noisy_images = max(1, int(current_batch_size * image_noise_percentage / 100))
            batch_dict["image_indices"] = sorted(random.sample(batch_indices, num_noisy_images))
        
        if text_noise_percentage > 0:
            num_noisy_texts = max(1, int(current_batch_size * text_noise_percentage / 100))
            batch_dict["text_indices"] = sorted(random.sample(batch_indices, num_noisy_texts))
        
        batch_noise_indices.append(batch_dict)
    
    with open(indices_path, 'w') as f:
        json.dump(batch_noise_indices, f)
    
    return batch_noise_indices


def noisy_chestx_collate_fn(batch, corruption_config=None, batch_idx=0, noise_indices=None):
    images = [sample[0][0] for sample in batch]
    reports = [sample[0][1] for sample in batch]
    targets = [sample[1] for sample in batch]
    
    images = torch.stack(images, dim=0)
    reports = torch.stack(reports, dim=0)  
    targets = torch.tensor(targets, dtype=torch.float)  
    if corruption_config is None or noise_indices is None:
        return reports, images, targets
    

    if batch_idx >= len(noise_indices):
        return reports, images, targets
    
    current_noise_indices = noise_indices[batch_idx]
    image_indices = current_noise_indices["image_indices"]
    text_indices = current_noise_indices["text_indices"]
    
    image_noise_type = corruption_config.get("image_noise_type")
    text_noise_type = corruption_config.get("text_noise_type")
    image_noise_params = corruption_config.get("image_noise_params", {})
    text_noise_params = corruption_config.get("text_noise_params", {})
    

    if image_noise_type and image_indices:
        if image_noise_type == 'gaussian':
            noisy_images = NoiseAugmenter.add_gaussian_noise_to_image(images, **image_noise_params)
        elif image_noise_type == 'salt_pepper':
            noisy_images = NoiseAugmenter.add_salt_and_pepper_noise(images, **image_noise_params)
        elif image_noise_type == 'poisson':
            noisy_images = NoiseAugmenter.add_poisson_noise(images, **image_noise_params)
        elif image_noise_type == 'quality_reduction':
            noisy_images = NoiseAugmenter.reduce_image_quality(images, **image_noise_params)
        elif image_noise_type == 'masking':
            noisy_images = NoiseAugmenter.mask_image_regions(images, **image_noise_params)
        
        for idx in image_indices:
            if idx < len(images): 
                images[idx] = noisy_images[idx]
    
    if text_noise_type and text_indices:
        if text_noise_type == 'dropout':
            noisy_reports = NoiseAugmenter.add_word_dropout(reports, **text_noise_params)
        elif text_noise_type == 'swap':
            noisy_reports = NoiseAugmenter.swap_words(reports, **text_noise_params)
        elif text_noise_type == 'corruption':
            noisy_reports = NoiseAugmenter.corrupt_text(reports, **text_noise_params)
        
        for idx in text_indices:
            if idx < len(reports): 
                reports[idx] = noisy_reports[idx]
    
    return reports, images, targets

class NoiseDataLoader(DataLoader):    
    def __init__(self, dataset, batch_size, shuffle, corruption_config, split_name, 
                 num_workers=0, collate_fn=None, seed=42, **kwargs):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.corruption_config = corruption_config
        self.split_name = split_name
        self.noise_seed = seed
        

        if corruption_config:
            self.noise_indices = get_noise_indices(
                dataset_size=len(dataset),
                batch_size=batch_size,
                config=corruption_config,
                split_name=split_name,
                seed=seed
            )
        else:
            self.noise_indices = None
        
        self.batch_idx = 0
        
        def noise_tracking_collate(batch):
            if collate_fn:
                result = collate_fn(batch, corruption_config, self.batch_idx, self.noise_indices)
                self.batch_idx = (self.batch_idx + 1) % ((len(dataset) + batch_size - 1) // batch_size)
                return result
            return torch.utils.data.dataloader.default_collate(batch)
        
        super(NoiseDataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=noise_tracking_collate,
            **kwargs
        )
    
    def __iter__(self):
        self.batch_idx = 0
        return super(NoiseDataLoader, self).__iter__()


def get_noisy_data_loaders(
    corruption_config=None,
    corruption_name=None,
    indices_path="split_indices.pth",
    data_path="./mm_health_bench/data/chestx",
    batch_size=32,
    num_workers=4,
    apply_to_train=True,
    apply_to_val=True,
    apply_to_test=True,
    max_seq_length=256,
    noise_seed=42
):
    random.seed(10)
    torch.manual_seed(10)
    config_to_use = CORRUPTIONS[corruption_config]
    
    from mm_health_bench.mmhb.loader import ChestXDataset
    from mm_health_bench.mmhb.utils import Config
    
    config = Config("./mm_health_bench/config/config.yml").read()
    chestx_dataset = ChestXDataset(data_path=data_path, max_seq_length=max_seq_length)
    
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
    
    
    train_loader = NoiseDataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        corruption_config=config_to_use if apply_to_train else None,
        split_name="train",
        num_workers=num_workers, 
        collate_fn=noisy_chestx_collate_fn,
        seed=noise_seed
    )
    
    val_loader = NoiseDataLoader(
        dataset=val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        corruption_config=config_to_use if apply_to_val else None,
        split_name="val",
        num_workers=num_workers, 
        collate_fn=noisy_chestx_collate_fn,
        seed=noise_seed
    )
    
    test_loader = NoiseDataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        corruption_config=config_to_use if apply_to_test else None,
        split_name="test",
        num_workers=num_workers, 
        collate_fn=noisy_chestx_collate_fn,
        seed=noise_seed
    )
    
    return train_loader, val_loader, test_loader


CORRUPTIONS = { "gaussian_25": {
            "name": "gaussian_25",
            "image_noise_type": "gaussian",
            "text_noise_type": None,
            "image_noise_params": {"mean": 0.0, "std": 0.1},
            "text_noise_params": None,
            "image_noise_percentage": 25,
            "text_noise_percentage": 0
        },
         "gaussian_50":{
            "name": "gaussian_50",
            "image_noise_type": "gaussian",
            "text_noise_type": None,
            "image_noise_params": {"mean": 0.0, "std": 0.1},
            "text_noise_params": None,
            "image_noise_percentage": 50,
            "text_noise_percentage": 0
        },
         "gaussian_75":{
            "name": "gaussian_75",
            "image_noise_type": "gaussian",
            "text_noise_type": None,
            "image_noise_params": {"mean": 0.0, "std": 0.1},
            "text_noise_params": None,
            "image_noise_percentage": 75,
            "text_noise_percentage": 0
        },
         "gaussian_100":{
            "name": "gaussian_100",
            "image_noise_type": "gaussian",
            "text_noise_type": None,
            "image_noise_params": {"mean": 0.0, "std": 0.1},
            "text_noise_params": None,
            "image_noise_percentage": 100,
            "text_noise_percentage": 0
        },
         "sp_0_25": {
            "name": "sp_0_25",
            "image_noise_type": "salt_pepper",
            "text_noise_type": None,
            "image_noise_params": {"amount": 0.25},
            "text_noise_params": None,
            "image_noise_percentage": 100,
            "text_noise_percentage": 0
        },
        "sp_0_50": {
            "name": "sp_0_50",
            "image_noise_type": "salt_pepper",
            "text_noise_type": None,
            "image_noise_params": {"amount": 0.5},
            "text_noise_params": None,
            "image_noise_percentage": 100,
            "text_noise_percentage": 0
        },
        "sp_0_75": {
            "name": "sp_0_75",
            "image_noise_type": "salt_pepper",
            "text_noise_type": None,
            "image_noise_params": {"amount": 0.75},
            "text_noise_params": None,
            "image_noise_percentage": 100,
            "text_noise_percentage": 0
        },
        "sp_0_100": {
            "name": "sp_0_100",
            "image_noise_type": "salt_pepper",
            "text_noise_type": None,
            "image_noise_params": {"amount": 1},
            "text_noise_params": None,
            "image_noise_percentage": 100,
            "text_noise_percentage": 0
        },
        "blur_1_5": {
            "name": "blur_1_5",
            "image_noise_type": "quality_reduction",
            "text_noise_type": None,
            "image_noise_params": {"blur_factor": 1.5},
            "text_noise_params": None,
            "image_noise_percentage": 100,
            "text_noise_percentage": 0
        },
        "blur_3": {
            "name": "blur_3",
            "image_noise_type": "quality_reduction",
            "text_noise_type": None,
            "image_noise_params": {"blur_factor": 3},
            "text_noise_params": None,
            "image_noise_percentage":  50,
            "text_noise_percentage": 0
        },
        "blur_5": {
            "name": "blur_5",
            "image_noise_type": "quality_reduction",
            "text_noise_type": None,
            "image_noise_params": {"blur_factor": 5},
            "text_noise_params": None,
            "image_noise_percentage": 100,
            "text_noise_percentage": 0
        },
        "blur_10": {
            "name": "blur_10",
            "image_noise_type": "quality_reduction",
            "text_noise_type": None,
            "image_noise_params": {"blur_factor": 10},
            "text_noise_params": None,
            "image_noise_percentage": 100,
            "text_noise_percentage": 0
        },

        "mask_01_2": {
            "name": "mask_01_2",
            "image_noise_type": "masking",
            "text_noise_type": None,
            "image_noise_params": {"mask_size": 0.1, "num_masks": 2},
            "text_noise_params": None,
            "image_noise_percentage":  50,
            "text_noise_percentage": 0
        },
        "mask_02_3": {
            "name": "mask_02_3",
            "image_noise_type": "masking",
            "text_noise_type": None,
            "image_noise_params": {"mask_size": 0.2, "num_masks": 3},
            "text_noise_params": None,
            "image_noise_percentage": 100,
            "text_noise_percentage": 0
        },
        "dropout_01":    {
            "name": "dropout_01",
            "image_noise_type": None,
            "text_noise_type": "dropout",
            "image_noise_params": None,
            "text_noise_params": {"dropout_prob": 0.1},
            "image_noise_percentage": 0,
            "text_noise_percentage": 100
        },
        "dropout_03": {
            "name": "dropout_03",
            "image_noise_type": None,
            "text_noise_type": "dropout",
            "image_noise_params": None,
            "text_noise_params": {"dropout_prob": 0.3},
            "image_noise_percentage": 0,
            "text_noise_percentage": 100
        },
        "dropout_05":{
            "name": "dropout_05",
            "image_noise_type": None,
            "text_noise_type": "dropout",
            "image_noise_params": None,
            "text_noise_params": {"dropout_prob": 0.5},
            "image_noise_percentage": 0,
            "text_noise_percentage": 100
        },
         "swap_01": {
            "name": "swap_01",
            "image_noise_type": None,
            "text_noise_type": "swap",
            "image_noise_params": None,
            "text_noise_params": {"swap_prob": 0.1},
            "image_noise_percentage": 0,
            "text_noise_percentage": 100
        },
        "swap_03": {
            "name": "swap_03",
            "image_noise_type": None,
            "text_noise_type": "swap",
            "image_noise_params": None,
            "text_noise_params": {"swap_prob": 0.3},
            "image_noise_percentage": 0,
            "text_noise_percentage": 100
        },
        "swap_05": {
            "name": "swap_05",
            "image_noise_type": None,
            "text_noise_type": "swap",
            "image_noise_params": None,
            "text_noise_params": {"swap_prob": 0.5},
            "image_noise_percentage": 0,
            "text_noise_percentage": 100
        },
        "corruption_01": {
            "name": "corruption_01",
            "image_noise_type": None,
            "text_noise_type": "corruption",
            "image_noise_params": None,
            "text_noise_params": {"corruption_prob": 0.1},
            "image_noise_percentage": 0,
            "text_noise_percentage": 100
        },
        "corruption_03": {
            "name": "corruption_03",
            "image_noise_type": None,
            "text_noise_type": "corruption",
            "image_noise_params": None,
            "text_noise_params": {"corruption_prob": 0.3},
            "image_noise_percentage": 0,
            "text_noise_percentage": 100
        },
        "corruption_05": {
            "name": "corruption_05",
            "image_noise_type": None,
            "text_noise_type": "corruption",
            "image_noise_params": None,
            "text_noise_params": {"corruption_prob": 0.5},
            "image_noise_percentage": 0,
            "text_noise_percentage": 100
        }
        }
