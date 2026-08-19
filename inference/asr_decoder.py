from pathlib import Path
import torch
from pyctcdecode import build_ctcdecoder
from dataset.asr_vocab import ID_TO_CHAR, VOCAB


# Beam search settings  
BEAM_WIDTH = 50
USE_LANGUAGE_MODEL = False

# Language model settings
alpha = 0.2
beta = 0


DECODER_LABELS = list(VOCAB)
DECODER_LABELS[0] = ""
PROJECT_ROOT = Path(__file__).resolve().parent.parent


LM_DIR = (
    PROJECT_ROOT
    / "language_models"
    / "speechtotext_en_us_lm_vdeployable_v4.1")


LM_PATH = LM_DIR / "3-gram.pruned.3e-7.arpa"


# Beam search WITHOUT KenLM
beam_decoder_no_lm = build_ctcdecoder(
    labels=DECODER_LABELS,)


# Beam search WITH KenLM
if USE_LANGUAGE_MODEL:
    if not LM_PATH.is_file():
        raise FileNotFoundError(
            f"Language model not found: {LM_PATH}")

    beam_decoder_with_lm = build_ctcdecoder(
        labels=DECODER_LABELS,
        kenlm_model_path=str(LM_PATH),
        alpha=alpha,
        beta=beta,)
else:
    beam_decoder_with_lm = None
    
    
def ctc_decode_beam(logits: torch.Tensor) -> str:
    
    """
    Performs CTC beam search by scoring competing transcript hypotheses across time steps,
    and selects the highest-scoring sequence. 
    
    Args:
        logits (torch.Tensor):
           Shape: (1, time_steps, 29)
        
    Returns:
        str: Decoded transcript.
    """    
        
    probs = (
        torch.softmax(logits[0], dim=-1)
        .detach()
        .cpu()
        .numpy())
    
    assert probs.shape[1] == len(DECODER_LABELS)
    
    if USE_LANGUAGE_MODEL:
        decoder = beam_decoder_with_lm
    else:
        decoder = beam_decoder_no_lm
    
    return decoder.decode(
        probs,
        beam_width=BEAM_WIDTH,)

    
def ctc_decode_greedy(predicted_ids: torch.Tensor) -> str:
    
    """
    Converts predicted CTC token IDs into a transcript by collapsing repeated
    consecutive tokens and removing blank tokens.
    
    Args:
        predicted_ids (torch.Tensor): Greedy CTC token predictions.
           Shape: (1, time_steps)
        
    Returns:
        str: Decoded transcript.
    """       
        
    ids = predicted_ids[0].tolist()

    collapsed = []
    prev = None
    
    # Loops through token IDs and collapses consecutive duplicate tokens. 
    for token in ids:
        if token != prev:
            collapsed.append(token)

        prev = token
     
    collapsed = [token for token in collapsed if token != 0]
    
    # Converts numerical IDs into strings and joins them. 
    return "".join( ID_TO_CHAR[token] for token in collapsed)