# Audio Transcription

## Project Abstract

This project implements a Bidirectional Long Short-Term Memory (Bi-LSTM) live-inference automatic speech recognition (ASR) system with a custom digital signal processing and feature extraction pipeline that converts raw audio waveforms into English text. 

The pipeline performs RNNoise-based denoising to suppress background noise, and uses voice activity detection to limit unnecessary ASR processing.  The resulting audio is converted to a log-Mel spectrogram and passed into a Bi-LSTM acoustic model trained using Connectionist Temporal Classification (CTC), producing character-level outputs decoded into corresponding text, using either greedy decoding or beam search with an optional KenLM language model. Text is displayed to the user via continuously updating partial transcripts, and a finalized transcript when silence is detected. 

The project explores whether Bi-LSTMs traditionally used for offline ASR tasks are adaptable into a live-inference system, using incremental, chunk-based audio processing. The system aims to retain the potential accuracy gains of bidirectional context while approaching the low latency of live-inference unidirectional systems.

## Results at a Glance

- **Model Size:** 617,757 trainable parameters
- **Training Data:** Approximately 600 hours of LibriSpeech (`train-clean-100` and `train-other-500`)
- **Greedy Decoding CER:** 26.6% on `train-other-500` validation data
- **Beam Search CER:** 25.5% on `train-other-500` validation data
- **Mean Partial Decode Latency:** 29.2 ms on Apple M4 Pro
- **Streaming Chunk Duration:** 100 ms

## Individual Contributions

### Nikhil Weerakoon: ASR / Machine Learning

- Implemented log-Mel feature extraction for ASR input
- Built and trained the Bi-LSTM acoustic model using CTC
- Implemented greedy and beam-search decoding
- Developed model training, evaluation, checkpointing, and CER measurement
- Designed and implemented the streaming inference pipeline
- Implemented online transcript post-processing and merging
- Wrote the project’s technical documentation and README

### Kyle Tran: Signal Processing

- Implemented RNNoise-based denoising
- Implemented voice activity detection
- Implemented audio resampling
- Implemented audio chunking prior to ASR feature extraction  

## Key Features

- **Streaming Audio Processing:** Processes live audio incrementally using fixed-size chunks rather than requiring the complete utterance.

- **RNNoise-Based Denoising:** Reduces background noise while preserving speech.

- **Voice Activity Detection:** Identifies whether a waveform contains speech and prevents unnecessary ASR processing of non-speech audio.

- **log-Mel Feature Extraction:** Extracts acoustic features from the waveform to construct a log-Mel spectrogram representing a perceptually emphasized frequency representation using windowing, FFT spectral analysis, and Mel filterbanks.

- **CTC-Based Training:** Uses Connectionist Temporal Classification to compute training loss without requiring frame-level alignment between audio features and their corresponding transcripts.

- **Feature Caching:** Precomputes and caches extracted log-Mel features during training to avoid recomputing them across repeated epochs.

- **Character Error Rate:** Evaluates model accuracy by determining the number of substitutions, deletions, and insertions required to convert the predicted transcript into the target transcript, normalized by the number of characters in the target.

- **Multiple Decoding Strategies:** Supports greedy decoding and beam search, with an optional language model scoring competing transcript hypotheses using linguistic context.

## System Architecture

During live inference, the microphone captures audio at 48 kHz; the raw waveform is processed through RNNoise-based denoising to suppress background noise. RNNoise returns a Voice Activity Detection (VAD) probability score measuring the probability that the waveform contains speech. If the score is at least 0.3, the audio is downsampled to 16 kHz and divided into 1600-sample (100 ms) chunks, then pushed onto a shared processing queue as a dictionary containing the event type and the processed waveform; otherwise, the audio is not further processed, and a dictionary indicating a silence event and its duration (100 ms) is pushed. 

Queued audio chunks accumulate in a rolling buffer for 48,000 samples (3 seconds); the buffered audio is then passed to ASR feature extraction. The buffered audio is then divided into 400-sample (25 ms) overlapping frames with a hop length of 160 samples (10 ms). A Hamming window is applied to each frame in preparation for the Fast Fourier Transform, reducing spectral leakage from boundary discontinuities. An FFT is applied to each frame, converting each frame from the time domain to the frequency domain with complex-valued frequency coefficients. The magnitude of each coefficient is computed to obtain a real-valued magnitude spectrogram. The spectrogram is projected into Mel-space with an 80-filter Mel filterbank, producing a perceptually emphasized frequency representation for model processing. A logarithmic transform is applied element-wise to the resulting Mel spectrogram, producing a log-Mel spectrogram for ASR processing. 

The log-Mel spectrogram is passed to a 2-layer Bi-LSTM recurrent neural network with 128 units per direction. Trained using a CTC objective, the network outputs a sequence of logit vectors over the model vocabulary. For decoding, either greedy decoding is used, selecting the maximum-scoring character at each time step before applying CTC collapse, or beam search is used to determine the highest-scoring transcript among competing CTC hypotheses, with an optional 3-gram KenLM language model incorporating linguistic history into scoring. 

The decoded output is displayed to the user via a continuously updating partial transcript every 300 ms, and a finalized transcript is produced after silence is detected for at least 700 ms.

```text
Raw Microphone Audio (48 kHz)
        ↓
RNNoise Denoising + Voice Activity Detection
        ↓
Downsampling (48 kHz → 16 kHz)
        ↓
Chunking (1,600 samples / 100 ms)
        ↓
Shared Processing Queue
        ↓
Overlapping Frame Extraction
(400 samples / 25 ms, 160-sample / 10 ms hop)
        ↓
Hamming Window + FFT
        ↓
Magnitude Spectrogram
        ↓
80-Filter Mel Filterbank + Log Transform
        ↓
Log-Mel Spectrogram
        ↓
2-Layer Bi-LSTM
(128 hidden units per direction)
        ↓
Character Logits / Probabilities
        ↓
Greedy Decoding / Beam Search + Optional KenLM
        ↓
Partial Transcription
        ↓
Transcript Merging
        ↓
Final Transcription
```
## Repository Structure

```text
audioTranscription/
│
├── dataset/
│   ├── asr_dataset.py          # Dataset loading and preparation
│   └── asr_vocab.py            # Character vocabulary and token mappings
│
├── docs/
│   └── results.md              # Training and evaluation results
│
├── features/
│   ├── asr_features.py         # log-Mel feature extraction
│   ├── audio_processing.py     # Audio preprocessing, denoising, VAD, and resampling
│   └── pre-compute-features.py # Precomputes and caches acoustic features for training
│
├── inference/
│   ├── asr_decoder.py          # CTC decoding utilities
│   └── decode.py               # Greedy and beam-search decoding
│
├── model/
│   └── asr_model.py.          # Bi-LSTM acoustic model definition
│                      
├── pipeline/
│   ├── offline.py              # Offline transcription pipeline
│   └── online.py               # Streaming transcription pipeline
│
├── postprocess/
│   └── onlineprocess.py        # Streaming transcript post-processing and merging
│
├── training/
│   ├── evaluate.py             # Model evaluation and CER measurement
│   ├── train.py                # Model training
│   └── utils.py                # Shared training utilities
│
├── diagnose_pyrnnoise.py       # Diagnostic utility for RNNoise
├── download_librispeech.py     # LibriSpeech dataset download utility
├── main.py                     # Main application entry point
├── requirements.txt            # Python dependencies
├── README.md
└── .gitignore
```
## Installation

The project is currently tested with Python 3.11. A virtual/Conda environment is recommended to keep dependencies isolated.

### Clone the Repository

```bash
git clone https://github.com/jnikhilw/audioTranscription 
cd audioTranscription
```

### Create and Activate an Environment

Using Conda:

```bash
conda create -n audio-transcription-env python=3.11
conda activate audio-transcription-env
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

KenLM is installed separately for language-model-assisted beam search:

```bash
pip install kenlm==0.2.0
```

On macOS, PyAudio may require PortAudio:

```bash
brew install portaudio
```
## Streaming Inference

Audio is captured through the microphone when main.py is run. Audio is processed incrementally through preprocessing and ASR pipelines, continuously displaying partial transcripts when speech is detected and finalized transcripts after an utterance is complete.

### Running the System

From the repository root:

```bash
python3 main.py
```

### Inference Configuration

| Parameter | Value |
|---|---|
| Checkpoint | train_other_500_best_val.pt |
| Default decoder | Greedy |
| Beam width | 50 |
| Language model | KenLM Language Model |

## Dataset

The Bi-LSTM acoustic model was trained sequentially on two subsets of the LibriSpeech speech recognition corpus:

- train-clean-100: approximately 100 hours of clean read speech.
- train-other-500: approximately 500 hours of more acoustically challenging read speech.

For each subset, 95% of the available samples were used for training, and 5% held out for validation.

| LibriSpeech Subset | Training Split | Validation Split |
|---|---|---|
| train-clean-100 | 95% | 5% |
| train-other-500 | 95% | 5% |

### Dataset Preparation

LibriSpeech was sampled at 16 kHz to match the live-inference feature extraction pipeline; transcripts are encoded with character-level lowercase letters, spaces, and apostrophes with the CTC blank token. Extracted log-Mel spectrograms were cached to disk to avoid redundant computation.

## Training

The Bi-LSTM acoustic model with 617,757 trainable parameters was trained with a Connectionist Temporal Classification (CTC) objective, which computes the negative log-likelihood of the sum of probabilities across all valid alignment paths that collapse to the target transcript to measure model loss. AdamW computed parameter updates with adaptive learning-rate scheduling applied during training. Learning occurred in two stages: first on the LibriSpeech train-clean-100-hour dataset for 30 epochs, followed by another 30 epochs on the more acoustically challenging train-other-500 dataset to improve generalization over acoustically varied environments.

We trained the model locally on the Apple M4 Pro with 24 GB of Unified Memory; operations unsupported by the MPS backend (CTC loss) were computed on the CPU, and the remaining operations on the MPS backend. Feature caching was used to avoid recomputing feature extraction over repeated epochs on the same dataset.

### Training Configuration

| Parameter | Value |
|---|---|
| Stage 1 dataset | train-clean-100 |
| Stage 1 epochs | 30 |
| Stage 2 dataset | train-other-500 |
| Stage 2 epochs | 30 |
| Training / validation split | 95% / 5% |
| Input features | 80 Mel bins |
| Bi-LSTM layers | 2 |
| Hidden size | 128 per direction |
| Batch size | 4 |
| Optimizer | AdamW |
| Initial learning rate | 5e-5 |
| Weight decay | 1e-4 |
| Gradient clipping | Max norm 1.0 |
| Learning-rate scheduler | ReduceLROnPlateau |
| Scheduler factor | 0.5 |
| Scheduler patience | 2 validation checks |
| Minimum learning rate | 1e-5 |
| Validation frequency | Every 3 epochs |
| Early-stopping patience | 5 validation checks |
| Compute mode | Apple MPS + CPU hybrid; CPU fallback |
| Trainable parameters | 617,757 |

### Optimization and Checkpointing

CTC loss is computed for each sample, normalized by the transcript length, and averaged across the batch. Gradients are clipped to 1.0, and CTC loss is checked for non-finite values before updating weights; if NaN or infinite values are detected, the training loop ceases to prevent corrupted weights. Weights are saved every epoch. Validation CER is measured every three epochs using greedy decoding, and model weights are saved to a separate best validation checkpoint if CER improves. If CER fails to improve for the configured patience threshold, ReduceLROnPlateau decreases the learning rate accordingly, with a minimum of 1e-5.

## Results

### Validation Performance

| Training Stage | Decoder | CER |
|---|---|---|
| train-clean-100 | Greedy | 0.19017317900365627 |
| train-other-500 | Greedy | 0.26563522868392525 |
| train-other-500 | Beam Search | 0.2554133292361512 |
| train-other-500 | Beam Search + KenLM | 0.2989078091388487 |

### Streaming Performance (Apple M4 Pro, Greedy Decoder)

| Metric | Value |
|---|---|
| Chunk duration | 100 ms |
| Mean partial decode latency | 29.2 ms |
| Median partial decode latency | 29.1 ms |
| Mean real-time factor (RTF) | 0.010 |
| Timing samples | 135 |

### Example Transcription

**Reference**

a work which fatigued me very much after this i went every day on board and brought away what i could get i had been now thirteen days on shore and had been eleven times on board the ship

**Prediction**

o wer quish the tery very much avfor this i went tevery day on bord and broght away wer i could get i had been now fertin days inshore an had biet aleven times onbor the ship

## Limitations and Future Work

While live-inference Bi-LSTMs using audio chunking allow the model to benefit from bidirectional phonetic context, the context is limited to each audio chunk rather than complete audio sequences, potentially reducing the ability to capture phonetic dependencies that extend beyond chunk boundaries. Importantly, the effectiveness of our model alone cannot empirically establish any comparative advantage over unidirectional systems for live-inference ASR, since a matched unidirectional model under otherwise identical conditions would need to be evaluated to establish a baseline. 

Secondly, validation performance tests show that while beam search improves CER, enabling the language model increases, rather than decreases, the character error rate; diagnosing and addressing the cause of this performance drop will be crucial to letting linguistic context influence predictions. 

Planned areas for improvement include:

- Further tuning beam-search decoding and KenLM integration.
- Further training on more acoustically challenging datasets.  
- Comparing the chunk-based Bi-LSTM against a matched unidirectional LSTM baseline.
- Reporting confidence intervals for Character Error Rate (CER).
- Completing an offline transcription pipeline.

## References

- A. Graves and J. Schmidhuber, “Framewise Phoneme Classification with Bidirectional LSTM and Other Neural Network Architectures,” Neural Networks, vol. 18, no. 5–6, pp. 602–610, 2005. doi: 10.1016/j.neunet.2005.06.042.
- A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber, “Connectionist Temporal Classification: Labeling Unsegmented Sequence Data with Recurrent Neural Networks,” Proceedings of the 23rd International Conference on Machine Learning (ICML), pp. 369–376, 2006. doi: 10.1145/1143844.1143891.
- V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “LibriSpeech: An ASR Corpus Based on Public Domain Audio Books,” 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 5206–5210, 2015. doi: 10.1109/ICASSP.2015.7178964.
- J.-M. Valin, “A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement,” 2018 IEEE 20th International Workshop on Multimedia Signal Processing (MMSP), pp. 1–5, 2018. doi: 10.1109/MMSP.2018.8547084.
- K. Heafield, “KenLM: Faster and Smaller Language Model Queries,” Proceedings of the Sixth Workshop on Statistical Machine Translation, pp. 187–197, 2011.
- Kensho Technologies, LLC, pyctcdecode: CTC Beam Search Decoder for Speech Recognition, version 0.5.0, 2023. GitHub repository: kensho-technologies/pyctcdecode.
- Z. Peng, PyRNNoise: Python Bindings for RNNoise, version 0.4.3, 2026.

## Acknowledgements

This project was developed collaboratively with Kyle Tran, who implemented the audio preprocessing pipeline, including RNNoise-based denoising, voice activity detection, resampling, and audio chunking. Nikhil Weerakoon implemented the ASR pipeline from feature extraction through acoustic modeling, CTC training, decoding, evaluation, streaming inference, transcript post-processing, and technical documentation.
