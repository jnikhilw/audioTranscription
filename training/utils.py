import torchaudio
import torch
import torchaudio


model_config = {
    "input_size": 80,
    "hidden_size": 256,
    "num_layers": 2,
    "vocab_size": 29,
}


def save_checkpoint(path, epoch, model, optimizer, avg_loss, best_val_cer=None):
    torch.save(
        {
            "epoch": epoch,
            "model_config": model_config,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "avg_loss": avg_loss,
            "best_val_cer": best_val_cer,
        },
        path
    )
    
    
def get_librispeech_split_range(  
    root: str,  
    url: str,  
    start_percent: float,  
    end_percent: float) -> tuple[int, int]:
    
    """ 
    Computes the starting index and sample size for the dataset subset specified by
    the start and end percent. 
    
    Args:  
        root (str): Root directory containing the LibriSpeech dataset.
        url (str): LibriSpeech subset to load.
        start_percent (float): Fractional starting position of the subset.
        end_percent (float): Fractional ending position of the subset.
        
    Returns:
        tuple[int, int]: Starting index and number of samples in the subset.
    """      
    
    full_dataset = torchaudio.datasets.LIBRISPEECH( 
        root=root,  
        url=url,  
        download=False  
    )
    
    total = len(full_dataset) 
    start = int(total * start_percent)    
    end = int(total * end_percent)
    
    limit = end - start
    
    return start, limit


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    
    """
    If non-finite values are detected in a tensor, print diagnostic statistics
    before raising an error to prevent corrupted gradients. 
     
    Args:  
        name (str): Name of the tensor being checked. 
        tensor (torch.Tensor): Tensor to check
        
    """        
    
    if not torch.isfinite(tensor).all():
        print(f"{name} contains non-finite values")
        print("shape:", tuple(tensor.shape))
        print("NaNs:", torch.isnan(tensor).sum().item())
        print("positive infs:", torch.isposinf(tensor).sum().item())
        print("negative infs:", torch.isneginf(tensor).sum().item())

        finite = tensor[torch.isfinite(tensor)]

        if finite.numel() > 0:
            print("finite min:", finite.min().item())
            print("finite max:", finite.max().item())
            print("finite mean:", finite.mean().item())

        raise RuntimeError(f"Non-finite values found in {name}")


def validate_ctc_batch(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> None:
    
    """
    Validates CTC tensor shapes, sequence lengths, and target token IDs before
    computing CTC loss.

    Args:
        log_probs (torch.Tensor): Log-probability tensor for CTC loss.
           Shape: (time_steps, batch_size, class_count)
        targets (torch.Tensor): Packed target token IDs for all samples in the batch.
        input_lengths (torch.Tensor): Number of valid model output time steps for each sample.
           Shape: (batch_size,)
        target_lengths (torch.Tensor): Number of target tokens for each sample.
           Shape: (batch_size,)
    
    """

    time_steps, batch_size, class_count = log_probs.shape

    if (
        input_lengths.numel() != batch_size
        or target_lengths.numel() != batch_size
    ):
        raise RuntimeError(
            "CTC lengths do not match the batch size."
        )

    if torch.any(input_lengths <= 0):
        raise RuntimeError(
            f"Non-positive input length: {input_lengths}"
        )

    if int(input_lengths.max()) > time_steps:
        raise RuntimeError(
            f"Input length exceeds model output length: "
            f"max_input={int(input_lengths.max())}, "
            f"time_steps={time_steps}"
        )

    if torch.any(target_lengths <= 0):
        raise RuntimeError(
            f"Non-positive target length: {target_lengths}"
        )

    if int(target_lengths.sum()) != targets.numel():
        raise RuntimeError(
            f"Packed target size mismatch: "
            f"sum(target_lengths)={int(target_lengths.sum())}, "
            f"targets.numel()={targets.numel()}"
        )

    if targets.numel() == 0:
        raise RuntimeError("Packed target tensor is empty.")

    minimum_target_id = int(targets.min())
    maximum_target_id = int(targets.max())

    # Blank is ID 0, so valid transcript IDs begin at 1.
    if minimum_target_id < 1 or maximum_target_id >= class_count:
        raise RuntimeError(
            f"Invalid target ID range "
            f"[{minimum_target_id}, {maximum_target_id}]. "
            f"Expected IDs from 1 to {class_count - 1}."
        )
    
    
def report_gradient_failure(
    model: torch.nn.Module,
    epoch: int,
    batch_index: int,
) -> None:
    """
    Reports the largest parameter gradient norms and non-finite gradient values
    after gradient clipping fails.

    Args:
        model (torch.nn.Module): Model whose parameter gradients are inspected.
        epoch (int): Training epoch where the failure occurred.
        batch_index (int): Batch index where the failure occurred.

    """

    reports = []

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue

        gradient = parameter.grad.detach()

        # CPU float64 reduces the chance that this diagnostic norm
        # calculation itself overflows.
        gradient_64 = gradient.cpu().double()

        reports.append(
            (
                torch.linalg.vector_norm(gradient_64).item(),
                name,
                gradient_64.abs().max().item(),
                torch.isnan(gradient_64).sum().item(),
                torch.isinf(gradient_64).sum().item(),
            )
        )

    reports.sort(reverse=True)

    print(
        f"\nGradient clipping failed: "
        f"epoch={epoch}, batch={batch_index}",
        flush=True,
    )

    for norm, name, max_abs, nan_count, inf_count in reports[:10]:
        print(
            f"{name}: "
            f"norm64={norm:.6e}, "
            f"max_abs={max_abs:.6e}, "
            f"NaNs={nan_count}, "
            f"infs={inf_count}",
            flush=True,
        )