import torch
from model.asr_model import ASRModel
from dataset.asr_dataset import load_audio_file
from features.asr_features import waveform_to_log_mel
from inference.asr_decoder import ctc_decode_greedy, ctc_decode_beam 


def load_model(checkpoint_path: str = "checkpoints/best_val.pt") -> ASRModel:
    
    """  
    Builds model architecture, loads model weights from the checkpoint path,
    and switches to evaluation mode. 
     
    Args:
        checkpoint_path (str): Path to the saved model checkpoint.
        
    Returns:
        model (ASRModel): Model with checkpoint weights ready for inference.       
    """
    
    # Build model architecture
    model = ASRModel()
    
    # Load model weights. 
    checkpoint = torch.load(   
        checkpoint_path,    
        map_location="cpu")
    
    model.load_state_dict(checkpoint["model_state"])
    
    # Switch to evaluation mode
    model.eval()
    
    return model 


def transcribe_features(features: torch.Tensor, model, decoder="greedy") -> str:
    
    """  
    Convert acoustic features to a logit score distribution and decode using either
    greedy decoding or beam search. 
     
    Args:
        features (torch.Tensor): Log-Mel spectrogram. 
        model (ASRModel): Model with checkpoint weights ready for inference.
        decoder (str): Decoding method, either "greedy" or "beam".
        
    Returns:
        str: Decoded transcript.    
    """    
    
    features = features.unsqueeze(0)

    with torch.no_grad():
        logits = model(features)

        if decoder == "greedy":
            prediction = logits.argmax(dim=2)
            text = ctc_decode_greedy(prediction)

        elif decoder == "beam":
            text = ctc_decode_beam(logits)

        else:
            raise ValueError(f"Unknown decoder: {decoder}")

    return text
