import torch
from dataset.asr_vocab import CHAR_TO_ID
from torch.utils.data import Dataset
from features.asr_features import waveform_to_log_mel, build_mel_filterbank
import torchaudio
import os


def normalize_transcript(text: str) -> str:

    """
    Standardizes raw text transcripts for character-level CTC tokenization.
    
    This function lowercases the input string and filters out non-vocabulary characters
    to prevent out-of-vocabulary (OOV) errors during model training and inference.
    The special "<blank>" token remains in the vocabulary but is excluded from transcript text.

    Args:
        text (str): Raw target transcript string. 

    Returns:
        str: Cleaned transcript containing only lowercase valid vocabulary characters.
    """
    
    text = text.lower()
    
    allowed_chars = set(CHAR_TO_ID.keys())
    allowed_chars.remove("<blank>")
    
    text = "".join(char for char in text if char in allowed_chars)
    
    return text


def text_to_ids(text: str) -> torch.Tensor:

    """
    Encodes a raw text transcript into a sequence of vocabulary integer IDs.

    This function executes the complete text preprocessing pipeline: it normalizes the input 
    string and maps each valid character key to its corresponding numerical value ID. The resulting 
    sequence is formatted as a rank 1 tensor.

    Args:
        text (str): Raw target transcript string.
    
    Returns:
        torch.Tensor: 1D tensor of token indices.
           Shape: (num_tokens,), Data Type: torch.long
    """
    
    text = normalize_transcript(text)
    ids = [CHAR_TO_ID[char] for char in text ]

    return torch.tensor(ids, dtype=torch.long)


def load_audio_file(audio_path: str) -> torch.Tensor:
    
    """
    Loads an audio asset and standardizes it into a single-channel 16kHz time-domain waveform.

    This function ensures structural uniformity across varying raw audio formats.
    It guarantees down-streaming components receive consistent sample lengths and channel shapes,
    preventing runtime errors in batch tensor operations.

    Args:
        audio_path (str): File path to the source audio file (e.g., .wav, .flac, .mp3).

    Returns:
        torch.Tensor: Normalized 1D temporal signal.
           Shape: (samples,), Data Type: torch.float32
    """ 
    
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert stereo to mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16k if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Remove channel dimension
    waveform = waveform.squeeze(0)

    return waveform


class LibriSpeechASRDataset(Dataset):
    """
      A custom PyTorch Dataset for loading acoustic-text pairs featuring a disk-caching mechanism.
      It includes an efficient file-system caching layer that stores extracted feature tensors on
      disk and reuses them across epochs, avoiding repeated DSP computation. 
      
      Args:
        Dataset : Librispeech acoustic dataset.
      """    
    

    def __init__(
        self,
        root: str = "data/librispeech",
        url: str = "dev-clean",
        start: int = 0,
        limit: int | None = None,
        use_cache: bool = True,         
        return_transcript: bool = False):
    
        """
        Initializes a LibriSpeech dataset subset and feature-extraction configuration.
    
        Args:
            root (str): Root directory containing the LibriSpeech dataset.
            url (str): LibriSpeech dataset subset to load.
            start (int): Starting subset index of the dataset. 
            limit (int): Number of samples to include after the starting index. 
            use_cache (bool): Determines if feature caching is enabled to store feature-extracted tensors. 
            return_transcript (bool): Determines whether to return transcripts for samples. 
               """        
        
        full_dataset = torchaudio.datasets.LIBRISPEECH(root=root, url=url, download=False)

        if limit is None:
            indices = range(start, len(full_dataset))
        else:
            indices = range(start, start + limit)

        self.dataset = torch.utils.data.Subset(full_dataset, indices)

        self.use_cache = use_cache
        self.cache_dir = os.path.join(
            "data",
            "cache",
            "librispeech",
            url)        

        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.mel_filterbank = build_mel_filterbank(sample_rate=16000, n_fft_bins=201, n_mels=80)
        
        self.return_transcript = return_transcript
        
    def __len__(self) -> int:
        
        """ Returns the total number of audio-transcript samples in the dataset."""
        
        return len(self.dataset)

    def __getitem__(self, idx):
        
        """ Fetches a dataset sample converts target transcripts into their token IDs and returns
        extracted features along with target IDs. If feature caching is enabled previously computed
        features are loaded from disk preventing recomputation, while features not computed are
        extracted and saved to disk. 
   
        
        Args:
            idx (int): index of sample in dataset.
            
        Returns:
            features (torch.Tensor): Log-mel spectrogram feature matrix.
               Shape: (num_frames, 80)
            target_ids (list): Target IDs corresponding to target transcripts.
            transcript (str, optional): Normalized transcript if return_transcript is enabled. 
        """
        waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = self.dataset[idx]

        cache_path = os.path.join(self.cache_dir, f"{speaker_id}_{chapter_id}_{utterance_id}.pt")

        target_ids = text_to_ids(transcript)

        if self.use_cache:
            try:
                features = torch.load(cache_path)
                if self.return_transcript:
                    return features, target_ids, normalize_transcript(transcript)
                return features, target_ids
            except FileNotFoundError:
                pass

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample( orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        waveform = waveform.squeeze(0).numpy()

        features = waveform_to_log_mel(waveform, self.mel_filterbank)

        if self.use_cache:
            torch.save(features, cache_path)
            
        if self.return_transcript:
            return features, target_ids, normalize_transcript(transcript)

        return features, target_ids

   
def collate_asr_batch(batch):
    
    """
    Collates a list of variable-length acoustic and text samples into structured, padded batch tensors.

    Because individual audio tracks and text transcripts vary in duration and length, this custom
    collator pads sequences to the maximum length within the current batch. It computes
    and tracks original sequence lengths, providing the temporal mapping required by CTC loss 
    functions (`nn.CTCLoss`) to ignore padding masks during optimization.

    Args:
        batch (list[tuple[torch.Tensor, torch.Tensor]]): A batch list containing individual sample tuples of
                                                        (log_mel_features, target_ids).

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
        padded_features (torch.Tensor): Padded feature tensor.
           Shape: (batch_size, max_num_frames, 80)
        padded_targets (torch.Tensor): Padded token ID tensor.
           Shape: (batch_size, max_num_tokens)
        input_lengths (torch.Tensor): 1D tensor of original acoustic frame counts.
           Shape: (batch_size,)
        target_lengths (torch.Tensor): 1D tensor of original transcript token counts.
           Shape: (batch_size,)
    """
    
    features = []
    targets = []
    
    # Separate audio features and transcript labels.    
    for x, y in batch:
    
        features.append(x)
        targets.append(y)
        
    # Store true lengths so CTC can ignore padded frames.
    input_lengths = torch.tensor(
        [x.shape[0] for x in features],
        dtype=torch.long
    )
    
    # Store true label lengths so CTC can ignore padded target IDs.
    target_lengths = torch.tensor(
        [y.shape[0] for y in targets],
        dtype=torch.long)
    
    # Pad feature sequences so they can be stacked into one tensor.
    padded_features = (
        torch.nn.utils.rnn
        .pad_sequence(
            features,
            batch_first=True))
    
    # Pad target sequences so all transcripts have the same length.
    padded_targets = (
        torch.nn.utils.rnn
        .pad_sequence(
            targets,
            batch_first=True))
    
    return (
        padded_features,
        padded_targets,
        input_lengths,
        target_lengths)