import numpy as np
import torch 


SAMPLE_Rate = 16_000
CHUNK_SIZE = 1600      # 100 ms, 10 chunks/sec
FRAME_SIZE = 400       # 25 ms
HOP_SIZE = 160         # 10 ms


def Chunk_to_Frames(
    chunk: np.ndarray,
    frame_size: int = FRAME_SIZE,
    hop_size: int = HOP_SIZE,
) -> np.ndarray:
    
    """
    Converts a 1D audio sample array into a frame matrix by
    dividing the chunk into overlapping acoustic frames, each
    corresponding to a time step. 
    
    Args:
        chunk (np.ndarray): 1D NumPy array of sampled audio.
           Shape: (chunk_size,). 
        frame_size (int): Number of samples per frame.
        hop_size (int): Sample distance separating each frame's starting sample. 
    
    Returns:
        np.ndarray: 2D matrix of framed audio samples
           Shape: (num_frames, frame_size). 
    """    
    
    chunk = np.asarray(chunk, dtype=np.float32)
    
    if chunk.size == 0:
        return np.empty((0, frame_size), dtype=np.float32,)
    
    # if the chunk size is less than the frame size, pad the chunk. 
    if chunk.size < frame_size:
        padding_needed = frame_size - chunk.size
        chunk = np.pad(
            chunk,
            pad_width = (0, padding_needed),
            mode = "constant",
            constant_values = 0,)
        
    frames = []
    
    # Loop across the chunk and append each frame to the frame list.  
    for i in range(0, len(chunk) - frame_size + 1, hop_size):   
        frame = chunk[i: i + frame_size]  
        frames.append(frame)
        
    return np.stack(frames).astype(np.float32, copy=False)

        
def remove_dc_offset(signal: np.ndarray)-> np.ndarray:
    
    """
    Removes the DC offset (0 Hz component) from the audio signal 
    by subtracting the global mean to center the waveform at 0.
    """
    
    return signal - np.mean(signal)


def apply_hamming_window(frames: np.ndarray) -> np.ndarray:
    
    """
    Applies a raised-cosine Hamming window to each audio frame,
    attenuating boundary amplitudes to reduce spectral leakage caused by
    discontinuities at frame edges before FFT/DFT computation.

    Args:
        frames np.ndarray: 2D matrix of framed audio samples.
        Shape: (num_frames, frame_size). 

    Returns:
        np.ndarray: 2D matrix of windowed audio frames with reduced edge discontinuities.
           Shape: (num_frames, frame_size)
    """
    
    frame_size = frames.shape[1]
    window = np.hamming(frame_size).astype(np.float32)
    return frames * window


def compute_spectrogram(frames: np.ndarray) -> np.ndarray:
    
    """""
    Computes the magnitude spectrogram of the input frames using a Real FFT.
    
    Applies a real-valued Fast Fourier Transform (FFT) to each of the 8 audio frames,
    converting each frame from the time domain to the frequency domain.
    Because each frame has 400 real-valued samples, rFFT returns 201 non-redundant
    frequency bins, including the DC component and the Nyquist-frequency bin.
    The complex Fourier coefficients are then converted to magnitudes.
    
    Args:
        frames np.ndarray: Frame matrix.
           Shape: (num_frames, frame_size).

    Returns:
        np.ndarray: Magnitude spectrogram.
           Shape: (num_frames, (frame_size/2)+1).      
    """
    
    # FFT along each row
    fft_result = np.fft.rfft(frames, axis=1)  
    # Convert complex Fourier coefficients to magnitudes
    magnitude = np.abs(fft_result)
    
    return magnitude


def hz_to_mel(frequency_hz: np.ndarray) -> np.ndarray:
    return 2595 * np.log10(1 + frequency_hz / 700)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700 * (10 ** (mel / 2595) - 1)


def build_mel_filterbank(
    sample_rate: int,
    n_fft_bins: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None, 
) -> np.ndarray:
    
    """  
    The filterbank maps linear-frequency FFT bins into perceptually
    spaced mel bins. Frequencies are first converted from Hz to mel,
    evenly subdivided in mel space, then converted back to Hz.
    This produces triangular filters that are narrower at low frequencies and wider at high frequencies.
    Applying this matrix compresses the FFT magnitude spectrum from many frequency bins
    into a smaller number of mel bands, emphasizing frequency resolution where human hearing is more sensitive.
    
    Args:
        sample_rate (int): Audio sample rate in Hz.
        n_fft_bins (int): Number of frequency bins from the rFFT.
        n_mels (int): Number of mel bands to create.
        f_min (float): Lowest frequency included in the filterbank.
        f_max (float | None): Highest frequency included. Defaults to Nyquist frequency.

    Returns:
        np.ndarray: Mel filterbank, each row is one triangular mel filter and each
        column corresponds to one FFT frequency bin.
           Shape: (n_mels, n_fft_bins).
    """

    if f_max is None:

        f_max = sample_rate / 2

    # Frequencies represented by FFT bins.
    fft_freqs = np.linspace(f_min, f_max, n_fft_bins)

    # Convert min/max Hz to mel.
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    # Need n_mels + 2 points because each triangle needs:
    
    # left edge, center, right edge.
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)

    # Convert those mel-spaced points back to Hz.
    hz_points = mel_to_hz(mel_points)

    # Create empty filterbank.
    filterbank = np.zeros((n_mels, n_fft_bins), dtype=np.float32)

    for m in range(1, n_mels + 1):
       
        left = hz_points[m - 1]
        center = hz_points[m]
        right = hz_points[m + 1]

        for k, freq in enumerate(fft_freqs):

            if left <= freq < center:
                filterbank[m - 1, k] = (freq - left) / (center - left)

            elif center <= freq < right:
                filterbank[m - 1, k] = (right - freq) / (right - center)

    return filterbank


def apply_mel_filterbank( magnitude_spec: np.ndarray, mel_filterbank: np.ndarray) -> np.ndarray:
    
    """ 
    Projects a linear-frequency magnitude spectrogram into mel space.
    This operation multiplies the magnitude spectrogram by the transpose
    of the mel filterbank matrix, aggregating neighboring FFT frequency
    bins into perceptually spaced mel bands. The transformation reduces spectral dimensionality
    while allocating greater frequency resolution to lower frequencies and less to higher frequencies,
    approximating human auditory perception.

    Args:
        magnitude_spec : np.ndarray
           Shape: (num_frames, n_fft_bins)
        mel_filterbank : np.ndarray
           Shape: (n_mels, n_fft_bins)

    Returns:
        np.ndarray: Mel spectrogram.
           Shape: (num_frames, n_mels)

    """
    mel_spec = magnitude_spec @ mel_filterbank.T
    return mel_spec.astype(np.float32, copy=False)


def compute_log_mel(mel_spec: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    
    """
    Applies a logarithmic transformation to the Mel Spectrogram element-wise.
    
    Human auditory perception responds approximately logarithmically to
    sound intensity rather than linearly. Applying the logarithm compresses
    large magnitude differences and expands smaller differences, producing
    a representation that better reflects perceived loudness.
    A small epsilon value is added before the logarithm to prevent undefined values from zero-energy mel bins.

    Args:
        mel_spec (np.ndarray): Mel spectrogram.
           Shape: (num_frames, n_mels)
        epsilon (float): Small numerical constant added for stability.

    Returns:
        np.ndarray: Log-mel spectrogram.
           Shape: (num_frames, n_mels)       
    """
    
    return np.log(mel_spec + epsilon).astype(np.float32, copy=False)


def waveform_to_log_mel(waveform: np.ndarray, mel_filterbank: np.ndarray) -> torch.Tensor:
    
    """
    Combines the feature pipeline functions into one function that converts a 1D audio NumPy array
    into an log-mel feature matrix for ASR processing. 
    
    Args:
        waveform (np.ndarray): 1D audio signal sampled at 16kHz.
           Shape: (samples,)
        mel_filterbank (np.ndarray): Precomputed Mel filterbank matrix.
           Shape: (80, n_fft_bins)
    
    Returns:
        torch.Tensor: Log-mel spectrogram feature matrix.
           Shape: (num_frames, 80)
    """
    
    waveform = remove_dc_offset(waveform)
    frames = Chunk_to_Frames(waveform)
    frames = apply_hamming_window(frames)
    spec = compute_spectrogram(frames)
    mel = apply_mel_filterbank(spec, mel_filterbank)
    log_mel = compute_log_mel(mel)

    return torch.tensor( log_mel, dtype=torch.float32)