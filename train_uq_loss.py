import torch
from torch import nn
import time
from eval_scripts.performance import f1_score
from tqdm import tqdm

def train_dynmm_multilabel(
    train_dataloader,
    valid_dataloader,
    moe_model,
    total_epochs,
    lr=1e-4,
    weight_decay=0.01,
    optimtype=torch.optim.AdamW,
    early_stop=True,
    objective=nn.BCEWithLogitsLoss(),
    save='best.pt',
    lambda_weight=0.01
):

    model = moe_model.cuda()
    
    optimizer = optimtype(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay
    )
    
    best_f1_macro = 0
    patience = 0
    patience_limit = 7  
    

    for epoch in range(total_epochs):
        model.train()
        total_loss = 0.0
        task_losses = 0.0
        resource_losses = 0.0
        total_samples = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            inputs = [i.cuda() for i in batch[:-1]]
            targets = batch[-1].cuda()
            
           
            outputs, fusion_weight = model(inputs)
            
            task_loss = objective(outputs, targets)
            
            quality_scores = model.batch_quality_scores
            uncertainty_scores = model.batch_uncertainty_scores
            branch_weights = model.batch_branch_weights
            
            per_sample_resource = torch.matmul(
                branch_weights, 
                model.flop.to(branch_weights.device)
            )
            
            quality_uncertainty_modifier = quality_scores * (1.0 - uncertainty_scores)
            
            resource_loss = per_sample_resource * quality_uncertainty_modifier.squeeze()
            mean_resource_loss = resource_loss.mean()
            
            total_batch_loss = task_loss + lambda_weight * mean_resource_loss
            
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 8.0)  # Gradient clipping
            optimizer.step()
            
            batch_size = targets.size(0)
            total_loss += total_batch_loss.item() * batch_size
            task_losses += task_loss.item() * batch_size
            resource_losses += mean_resource_loss.item() * batch_size
            total_samples += batch_size
            
            progress_bar.set_postfix({
                'loss': total_loss / total_samples,
                'task_loss': task_losses / total_samples,
                'resource_loss': resource_losses / total_samples
            })
        
        model.eval()
        model.reset_weight()  
        
        val_loss = 0.0
        val_samples = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in valid_dataloader:
                inputs = [i.cuda() for i in batch[:-1]]
                targets = batch[-1].cuda()
                
                outputs, _ = model(inputs)
                
                loss = objective(outputs, targets)
                
                batch_size = targets.size(0)
                val_loss += loss.item() * batch_size
                val_samples += batch_size
                
                all_preds.append(torch.sigmoid(outputs).round())
                all_targets.append(targets)
        
        branch_stats = model.get_selection_stats() if hasattr(model, 'get_selection_stats') else "Stats not available"
        branch_weights = model.weight_stat() if hasattr(model, 'weight_stat') else 0
        
        all_preds = torch.cat(all_preds, 0)
        all_targets = torch.cat(all_targets, 0)
        
        f1_micro = f1_score(all_targets, all_preds, average="micro")
        f1_macro = f1_score(all_targets, all_preds, average="macro")
        
        print('-' * 70)
        print(f'Epoch {epoch+1}/{total_epochs}:')
        print(f'Train loss: {total_loss/total_samples:.4f} | '
              f'Task loss: {task_losses/total_samples:.4f} | '
              f'Resource loss: {resource_losses/total_samples:.4f}')
        print(f'Val loss: {val_loss/val_samples:.4f} | '
              f'F1 micro: {f1_micro:.4f} | F1 macro: {f1_macro:.4f}')
        print(f'Branch weights: {branch_weights:.4f}')
        print(branch_stats)
        
        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            patience = 0
            print(f"New best F1 macro: {best_f1_macro:.4f}, saving model to {save}")
            torch.save(model, save)
        else:
            patience += 1
            print(f"No improvement, patience: {patience}/{patience_limit}")
            
        if early_stop and patience >= patience_limit:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break
    
    print(f"Training completed. Best F1 macro: {best_f1_macro:.4f}")
    return model


def test_dynmm_multilabel(model, test_dataloader, objective=nn.BCEWithLogitsLoss()):
    model.eval()
    model.reset_weight()  
    model.reset_selection_stats()
    
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            inputs = [i.cuda() for i in batch[:-1]]
            targets = batch[-1].cuda()
            
            outputs, _ = model(inputs)
            
            loss = objective(outputs, targets)
            
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            all_preds.append(torch.sigmoid(outputs).round())
            all_targets.append(targets)
    
    all_preds = torch.cat(all_preds, 0)
    all_targets = torch.cat(all_targets, 0)
    
    f1_micro = f1_score(all_targets, all_preds, average="micro")
    f1_macro = f1_score(all_targets, all_preds, average="macro")
    
    branch_stats = model.get_selection_stats()
    fusion_weight = model.weight_stat()
    flops = model.cal_flop()
    
    print('-' * 70)
    print('Test Results:')
    print(f'Loss: {total_loss/total_samples:.4f} | '
          f'F1 micro: {f1_micro:.4f} | F1 macro: {f1_macro:.4f}')
    print(f'Average branch fusion weight: {fusion_weight:.4f}')
    print(f'Effective FLOPs: {flops:.2f}M')
    print(branch_stats)
    
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'loss': total_loss/total_samples,
        'fusion_weight': fusion_weight,
        'flops': flops
    }
