import queue
import threading
import numpy as np
from features.asr_features import build_mel_filterbank, waveform_to_log_mel
from inference.decode import transcribe_features
from features.audio_processing import stream_48k_file_to_pipeline, stream_microphone_to_pipeline
from postprocess.onlineprocess import postprocess_online

SAMPLE_RATE = 16_000
DECODE_WINDOW_SAMPLES = 48000  # 3 seconds
DECODE_STRIDE_SAMPLES = 4800 
FINALIZE_AFTER_SILENCE_MS = 700


def decode_audio(audio: np.ndarray, mel_filterbank, model) -> str:

    """Convert one buffered 16 kHz waveform into cleaned transcript text."""

    if len(audio) == 0:
        return ""

    features = waveform_to_log_mel(audio, mel_filterbank)
    prediction = transcribe_features(
        features,
        model,
        decoder="beam"
    )    
    return postprocess_online(prediction)


def run_online(input_file: str | None , model):

    """
    Run online transcription using either:
    - live microphone input when input_file is None
    - simulated streaming from a 48 kHz WAV file otherwise
    
    Queue event contract:

    Speech:
        {"type": "speech",
        "audio": np.ndarray

    Silence:
        {"type": "silence",
        "duration_ms": 100}

    End:
        {"type": "end"}

    """
    pipeline_queue = queue.Queue(maxsize=100)
    
    mel_filterbank = build_mel_filterbank(
        sample_rate= SAMPLE_RATE,
        n_fft_bins=201,
        n_mels=80)
    
    if input_file is None:
        producer_target = stream_microphone_to_pipeline
        producer_args = (pipeline_queue, )
    else:
        producer_target = stream_48k_file_to_pipeline
        producer_args = (input_file,pipeline_queue)

    audio_buffer = np.empty(0, dtype=np.float32)
    
    accumulated_silence_ms = 0
    
    audio_thread = threading.Thread(
        target= producer_target,
        args= producer_args,
        daemon=True
    )
    

    audio_thread.start()
    
    print("Starting online ASR pipeline...")

    while True:
        
        event = pipeline_queue.get()
        event_type = event.get("type")

        if event_type == "speech":
            audio = event["audio"]
            
                   
            
            #sample_rate = event["sample_rate"]

            #if sample_rate != SAMPLE_RATE:
                #raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sample_rate} Hz.")

            audio_buffer = np.concatenate((audio_buffer, audio))

            # Speech has resumed, so reset the consecutive-silence counter.
            accumulated_silence_ms = 0

            while len(audio_buffer) >= DECODE_WINDOW_SAMPLES:

                window = audio_buffer[:DECODE_WINDOW_SAMPLES]
                prediction = decode_audio(
                    window,
                    mel_filterbank,
                    model)

                if prediction:
                    print("partial transcript:", prediction)

                # Advance 300 ms while retaining overlapping context.

                audio_buffer = audio_buffer[DECODE_STRIDE_SAMPLES:]

        elif event_type == "silence":

            accumulated_silence_ms += event.get("duration_ms", 0)

            if accumulated_silence_ms >= FINALIZE_AFTER_SILENCE_MS:
                if len(audio_buffer) > 0:
                    final_prediction = decode_audio(
                        audio_buffer,
                        mel_filterbank,
                        model)

                    if final_prediction:
                        print("final transcript:", final_prediction)

                # Begin a fresh utterance after sustained silence.
                
                audio_buffer = np.empty(0, dtype=np.float32)
                accumulated_silence_ms = 0

        elif event_type == "end":
            
            # Decode speech remaining after the last full window.
            if len(audio_buffer) > 0:

                final_prediction = decode_audio(
                    audio_buffer,
                    mel_filterbank,
                    model)

                if final_prediction:
                    print("final transcript:", final_prediction)

            break

        else:
            print("Ignoring unknown queue event:", event)

    audio_thread.join()
    print("Online ASR pipeline finished.")