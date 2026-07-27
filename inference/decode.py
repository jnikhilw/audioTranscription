import torch
from model.asr_model import ASRModel
from dataset.asr_dataset import (load_audio_file, ASRDataset)
from features.asr_features import waveform_to_log_mel
from inference.asr_decoder import ctc_decode_greedy, ctc_decode_beam 
import torch.nn.functional as F


def load_model(checkpoint_path: str = "checkpoints/best_val.pt" ) -> ASRModel:
    # Build model architecture
    model = ASRModel()
    
    checkpoint = torch.load(   
        checkpoint_path,    
        map_location="cpu")
    
    model.load_state_dict(checkpoint["model_state"])
    
    # Switch to inference mode
    model.eval()
    
    return model 

  
def transcribe_audio(audio_path: str , model: ASRModel) -> str:
    
    waveform = load_audio_file(audio_path)
    waveform = waveform.numpy()
    
    dataset = ASRDataset([])
    
    features = waveform_to_log_mel(waveform, dataset.mel_filterbank)
    features = features.unsqueeze(0)
    
    with torch.no_grad():
        logits = model(features)
        
        probs = torch.softmax(logits, dim=2)
        
        top_probs, top_ids = probs[0].max(dim=1)
        
        print("num nonblank frames:", (top_ids != 0).sum().item())
        print("blank prob mean:", probs[0, :, 0].mean().item())
        print("blank prob min:", probs[0, :, 0].min().item())
        print("first nonblank ids:", top_ids[top_ids != 0][:50])
        
        prediction = logits.argmax(dim=2)
        text = ctc_decode_greedy(prediction)
    return text 
    

def transcribe_features(features, model, decoder="greedy"):
    

    features = features.unsqueeze(0)

    with torch.no_grad():
        logits = model(features)
        

        if decoder == "greedy":
            prediction = logits.argmax(dim=2)
            text = ctc_decode_greedy(prediction)

        elif decoder == "beam":
            text = ctc_decode_beam(logits)

        else:
            raise ValueError(
                f"Unknown decoder: {decoder}"
            )

    return text
