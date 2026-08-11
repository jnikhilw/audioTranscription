# audioTranscription

DRAFT

## Project Abstract 
This project implements a Bi-LSTM live-inference automatic speech recognition (ASR) that converts raw audio waveforms into corresponding transcripts. The pipeline performs RNNoise-based-denoising to suppress background noise and voice-activity detection to determine whether an incoming signal requires ASR processing. A log-Mel spectrogram is then extracted from the resulting waveform and passed as input into the Bi-LSTM acoustic model trained using Connectionist Temporal Classification (CTC). The resulting probability distributions are decoded into transcripts using either greedy decoding or beam search with an optional KenLM language model. 
This project explores whether Bi-LSTMs traditionally used for offline ASR tasks are adaptable into a live-inference system using incremental, chunk-based processing on incoming audio. The system aims to retain the potential accuracy gains of bidirectional context while approaching the low-latency of live-inference unidirectional systems. 

## Key Features
Streaming audio processing: Processes live audio incrementally using fixed-size chunks rather than waiting for the full signal.

RNNoise-denoising: Reduces background noise while preserving speech. 

Voice-Activity-Detection: Identifies speech-containing regions of the audio signal and prevents ASR processing of non-speech signals. 

Log-Mel feature extraction: Extracts features from the audio signal to construct a log-Mel spectrogram that represents perceptually relevant frequency information using windowing, FFT spectral analysis, and Mel filterbanks.

CTC-based training: Uses Connectionist Temporal Classification to compute loss, without requiring frame-level alignment between audio features and the corresponding transcripts. 

Feature caching: Precomputes and caches extracted log-Mel features during training to avoid recomputing them in later epochs.

Character Error Rate: Evaluates model accuracy by determining the number of substitutions, deletions, and insertions required to convert the predicted transcript into the target transcript, normalized by the number of characters in the target. 

Multiple Decoding Strategies: Supports greedy decoding, selecting the max-scoring character across logits at each step before applying CTC collapse, and beam search with an optional language model that considers competing transcripts and linguistic context. 

## System Architecture

During live inference, the microphone records audio at 48 kHz; the signal is then processed through RNNoise-based denoising to suppress background noise. RNNoise then returns a VAD probability score measuring the probability that the signal contains speech. If the score is greater than 0.3, the audio is downsampled to 16 kHz and divided into 1600-sample (100 ms) fixed-size chunks, then pushed onto a shared processing queue; otherwise, the pipeline does not push the waveform onto the queue, preventing unnecessary ASR processing. Queued audio chunks accumulate in a rolling buffer until [ms of audio] is collected, which the pipeline then passes onto ASR processing. Each queued chunk is divided into eight 400-sample (25 ms) overlapping frames with a hop length of 160 samples (10 ms). A Hamming window is applied to each frame in preparation for FFT application, reducing spectral leakage from boundary discontinuities. An FFT is applied to each frame, converting each frame from the time domain into the frequency domain with complex-valued frequency coefficients. The magnitude of each coefficient is then computed to obtain a real-valued magnitude spectrogram. The spectrogram is then projected onto mel-space with an 80-filter mel filterbank, producing a perceptually relevant frequency representation for the ASR to process. A logarithmic transformation is applied to output a log-Mel spectrogram. The spectrogram is passed to a 2-layer bidirectional Long Short-Term Memory network with 128 units per direction. Trained using Connectionist Temporal Classification, the network outputs a sequence of logit vectors covering all characters in the vocabulary; for decoding, either greedy decoding is used to select the character corresponding to the maximum logit at each time-step before applying CTC collapse, or beam search with 50 width is used to determine the highest scoring transcript among competing CTC hypotheses, along with a 3-gram KenLM language model that accounts for linguistic history. The decoded output is displayed as partial transcripts; then partial transcripts are merged and output as a final transcript.

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
2-Layer BiLSTM
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

## Repository Structure

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
│   ├── asr_features.py         # Log-Mel feature extraction
│   ├── audio_processing.py     # Audio preprocessing, denoising, VAD, and resampling
│   └── pre-compute-features.py # Precomputes and caches acoustic features for training
│
├── inference/
│   ├── asr_decoder.py          # CTC decoding utilities
│   └── decode.py               # Greedy and beam-search decoding
│
├── metadata/                   # [description]
│
├── model/
│   ├── asr_engine.py           # [description]
│   └── asr_model.py            # BiLSTM acoustic model definition
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

## Installation

The project is currently tested with Python 3.11. A virtual/Conda environment is recommended to keep dependencies isolated.

Clone the Repository
git clone https://github.com/jnikhilw/audioTranscription 
cd audioTranscription

Create and Activate an Environment
Using Conda:
conda create -n audio-transcription-env python=3.11
conda activate audio-transcription-env

Install Dependencies
pip install -r requirements.txt
KenLM is installed separately for language-model-assisted beam search:
pip install kenlm==0.2.0
On macOS, PyAudio may require PortAudio:
brew install portaudio

## Streaming Inference
Audio is captured through the microphone when main.py is run. Audio is processed incrementally through pre-processing and ASR pipelines and outputting corresponding text if speech is detected.
Running the System
From the repository root:
python3 main.py

# Inference Configuration
Parameter
Value
Checkpoint
train_other_500_best_val.pt 
Default decoder
Greedy
Beam width
50
Language model
KenLM Language Model


## Dataset
The Bi-LSTM acoustic model  trained on two subsets of the LibriSpeech speech recognition corpus:
train-clean-100:  approximately 100 hours of clean read speech.
train-other-500: approximately 500 hours of more acoustically challenging read speech.

For each subset, 95% of the available samples  were used for training, and 5% held out for validation.
LibriSpeech Subset
Training Split
Validation Split
train-clean-100
95%
5%
train-other-500
95%
5%

We trained the model sequentially across the two subsets: initial training was performed on train-clean-100, after which training continued from the resulting checkpoint on train-other-500.

Dataset Preparation 
LibriSpeech is sampled at 16 kHz to match live-inference feature extraction; transcripts are encoded with character-level lowercase letters, spaces, and apostrophes with the CTC blank token. Extracted log-Mel spectrograms are cached to disk to avoid unnecessary computation.

## Training
The Bi-LSTM acoustic model with 128 units per direction and 617,757 trainable parameters was trained using Connectionist Temporal Classification (CTC), which computes the negative log-likelihood of the sum of the probabilities across all frame-level alignment paths that collapse to the target transcript to measure model loss. AdamW computed parameter updates with adaptive learning-rate scheduling. Learning occurred in two stages: first on the LibriSpeech train-clean-100-hour dataset for thirty epochs, followed by another 30 epochs on the more acoustically challenging train-other-500 dataset to improve generalization over acoustically varied environments. We trained the model locally on the Apple M4 Pro; operations unsupported by the MPS backend (CTC loss) were computed on the CPU, and the rest on the MPS GPU backend. Feature caching was used to avoid recomputing feature extraction over later epochs on the same dataset. 

Training Configuration
Parameter
Value
Stage 1 dataset
train-clean-100
Stage 1 epochs
30
Stage 2 dataset
train-other-500
Stage 2 epochs
30
Training / validation split
95% / 5%
Input features
80 Mel bins
BiLSTM layers
2
Hidden size
128 per direction
Batch size
4
Optimizer
AdamW
Initial learning rate
5e-5
Weight decay
1e-4
Gradient clipping
Max norm 1.0
Learning-rate scheduler
ReduceLROnPlateau
Scheduler factor
0.5
Scheduler patience
2 validation checks
Minimum learning rate
1e-5
Validation frequency
Every 3 epochs
Early-stopping patience
5 validation checks
Compute mode
Apple MPS + CPU hybrid; CPU fallback
Trainable parameters
617,757

Optimization and Checkpointing 
CTC loss is computed for each sample, normalized by the transcript length, and averaged across the batch size. Gradients are clipped to 1.0, and CTC loss is checked for non-finite values before updating; if NaN or infinite values are detected, the training loop ceases to avoid corrupted weights. Weights are saved every epoch. Validation CER is measured every three epochs using greedy decoding and saved to a separate best validation checkpoint. If CER fails to improve for the configured patience threshold, ReduceLROnPlateau decreases the learning rate accordingly, with a minimum of 1e-5.

## Results
Validation Performance
Training Stage
Decoder
CER
train-clean-100
Greedy
0.19017317900365627
train-other-500
Greedy
0.26563522868392525
train-other-500
Beam Search
0.2554133292361512
train-other-500
Beam Search + KenLM
0.2989078091388487

Streaming Performance
Metric
Value
Chunk duration
100 ms
Average inference latency
[latency]
Real-time factor (RTF)
[RTF]

Example Transcription
Reference
truth: a work which fatigued me very much after this i went every day on board and brought away what i could get i had been now thirteen days on shore and had been eleven times on board the ship
Prediction
o wer quish the tery very much avfor this i went tevery day on bord and broght away wer i could get i had been now fertin days inshore an had biet aleven times onbor the ship

## Limitations and Future Work
While live-inference Bi-LSTMs using audio chunking allow the model to benefit from bidirectional context, the context is limited to individual audio chunks rather than complete audio sequences, potentially reducing its effectiveness in capturing phonetic dependencies that extend beyond chunk boundaries.  Importantly, the effectiveness of our model alone cannot empirically attest to any comparative benefit over unidirectional systems in live inference, since a matched unidirectional model under otherwise identical conditions will need to be evaluated to establish a baseline. 

Planned areas for improvement include.
Further tuning beam-search decoding and KenLM integration.
Comparing the chunk-based BiLSTM against a unidirectional LSTM baseline.
Reporting confidence intervals for Character Error Rate.
Completing an offline transcription pipeline

## References

A. Graves and J. Schmidhuber, “Framewise Phoneme Classification with Bidirectional LSTM and Other Neural Network Architectures,” Neural Networks, vol. 18, no. 5–6, pp. 602–610, 2005. doi: 10.1016/j.neunet.2005.06.042.
A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber, “Connectionist Temporal Classification: Labeling Unsegmented Sequence Data with Recurrent Neural Networks,” Proceedings of the 23rd International Conference on Machine Learning (ICML), pp. 369–376, 2006. doi: 10.1145/1143844.1143891.
V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “LibriSpeech: An ASR Corpus Based on Public Domain Audio Books,” 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 5206–5210, 2015. doi: 10.1109/ICASSP.2015.7178964.
J.-M. Valin, “A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement,” 2018 IEEE 20th International Workshop on Multimedia Signal Processing (MMSP), pp. 1–5, 2018. doi: 10.1109/MMSP.2018.8547084.
K. Heafield, “KenLM: Faster and Smaller Language Model Queries,” Proceedings of the Sixth Workshop on Statistical Machine Translation, pp. 187–197, 2011.
Kensho Technologies, LLC, pyctcdecode: CTC Beam Search Decoder for Speech Recognition, version 0.5.0, 2023. GitHub repository: kensho-technologies/pyctcdecode.
Z. Peng, PyRNNoise: Python Bindings for RNNoise, version 0.4.3, 2026.


