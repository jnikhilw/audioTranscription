import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.asr_model import ASRModel
from dataset.asr_dataset import LibriSpeechASRDataset, collate_asr_batch
import os
from training.utils import save_checkpoint ,  get_librispeech_split_range, validate_ctc_batch, report_gradient_failure 
import time
from training.evaluate import eval_diagnostics,  eval_dataset_val

    
os.makedirs("checkpoints", exist_ok=True)

# Compute Device Allocation
mode = "hybrid"  # "cpu" or "hybrid"

# Save-settings
use_cache = True
load_progress = True
load_best = False

# Train-set range
start_percent = 0.0
end_percent = 0.95

# Training length
start_epoch = 0
num_epoch = 81

# Optimization-settings
batch_size = 4
num_workers = 0
shuffle = True
learn_rate = 5e-5
weight_decay = 1e-4
clip_grad_maxnorm = 1.0

# etc
best_val_cer = None


if mode == "hybrid" and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("mode:", mode)
print("device:", device)


start, limit =  get_librispeech_split_range(root="data/librispeech",
    url="train-other-500",
    start_percent= start_percent,
    end_percent= end_percent)


model = ASRModel().to(device)


dataset = LibriSpeechASRDataset(
    root="data/librispeech",
    url="train-other-500",
    start= start,
    limit= limit,
    use_cache=use_cache)


loader = DataLoader(
    dataset,
    batch_size = batch_size,
    shuffle = shuffle,
    num_workers = num_workers, 
    collate_fn=collate_asr_batch)


print("dataset size:", len(dataset))


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = learn_rate,
    weight_decay = weight_decay)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=1e-5)


if load_progress:
    if load_best and os.path.exists('checkpoints/best_val.pt'):
        checkpoint = torch.load("checkpoints/best_val.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        # optimizer.load_state_dict(checkpoint["optimizer_state"])
        # best_val_cer = checkpoint["best_val_cer"]
        # start_epoch = checkpoint["epoch"] + 1           
        print("loaded best existing checkpoint")
        
    elif os.path.exists('checkpoints/train_other_500_latest.pt'):
        checkpoint = torch.load("checkpoints/train_other_500_latest.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_val_cer = checkpoint["best_val_cer"]
        start_epoch = checkpoint["epoch"] + 1        
        print("loaded latest existing checkpoint")
        

ctc = nn.CTCLoss(blank=0, reduction = "none", zero_infinity=False)


start_time = time.time()
patience_counter = 0

for epoch in range(start_epoch, num_epoch):
    
    model.train()

    total_loss = 0.0

    for batch_index, (features, targets, input_lengths, target_lengths,) in enumerate(loader):
        
        features = features.to(device)
          
        targets = torch.cat([ target[:int(length)] for target, length in zip( targets, target_lengths,)
        ]).to(dtype=torch.long)
        
        optimizer.zero_grad(set_to_none=True)
        
        logits = model(features)
        
        log_probs = F.log_softmax(logits, dim=2)
        log_probs = log_probs.permute(1, 0, 2)
        
                   
        validate_ctc_batch(
            log_probs,
            targets,
            input_lengths,
            target_lengths,)
        
        if mode == "hybrid":
            per_sample_losses = ctc(
                log_probs.cpu(),
                targets.cpu(),
                input_lengths.cpu(),
                target_lengths.cpu(),)
            
        else:
            per_sample_losses = ctc(
                log_probs,
                targets,
                input_lengths,
                target_lengths,)
        
        
        if not torch.isfinite(per_sample_losses).all():
            bad_indices = torch.where(
                ~torch.isfinite(per_sample_losses)
            )[0].tolist()
            
            raise RuntimeError(
                f"Non-finite CTC loss at epoch={epoch}, "
                f"batch={batch_index}, "
                f"bad_items={bad_indices}, "
                f"input_lengths={input_lengths.tolist()}, "
                f"target_lengths={target_lengths.tolist()}"
            )
        
        
        loss_target_lengths = target_lengths.to(
            device=per_sample_losses.device,
            dtype=per_sample_losses.dtype,)
        
        loss = ( per_sample_losses / loss_target_lengths).mean()
        

        
        loss.backward()
        

        
        try:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=clip_grad_maxnorm,
                error_if_nonfinite=True,
            )
        
        except RuntimeError:
            report_gradient_failure(
                model,
                epoch,
                batch_index,
            )
            raise


        optimizer.step()
        
        if batch_index % 100 == 0:
            print(
                f"epoch={epoch}, "
                f"batch={batch_index}, "
                f"loss={loss.item():.4f}, "
                f"grad_norm={gradient_norm.item():.4f}",
                flush=True,
            )        
        
        total_loss += loss.item()
        

    avg_loss = total_loss / len(loader)
    print(f'Epoch {epoch}:',f'loss {avg_loss}') 


    save_checkpoint(
        "checkpoints/train_other_500_latest.pt",
        epoch,
        model,
        optimizer,
        avg_loss,
        best_val_cer)
    
    print("saved checkpoint")
        
    if epoch % 3 == 0:
            
        current_cer = eval_diagnostics(eval_dataset_val, inspect_predictions = False, skill_score = False, DECODER = "greedy")
            
        scheduler.step(current_cer)
            
        current_lr = optimizer.param_groups[0]["lr"]
            
        print("current learning rate:", current_lr)
        
        
        if best_val_cer == None:
            best_val_cer = current_cer
            save_checkpoint(
                "checkpoints/train_other_500_best_val.pt",
                epoch,
                model,
                optimizer,
                avg_loss,
                best_val_cer)
            print("saved best checkpoint")
                
                
        elif current_cer < best_val_cer:
            best_val_cer = current_cer
            save_checkpoint(
                "checkpoints/train_other_500_best_val.pt",
                epoch,
                model,
                optimizer,
                avg_loss,
                best_val_cer)
            print("saved new best checkpoint")
            patience_counter = 0
                
        else:
            patience_counter += 1
            print(f'patience count: {patience_counter}')
            if patience_counter >= 5:
                break 
            
            
end_time = time.time()     
print("total training time:", end_time - start_time)


