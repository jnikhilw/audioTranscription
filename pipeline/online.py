import queue
import threading
import numpy as np
from features.asr_features import build_mel_filterbank, waveform_to_log_mel
from inference.decode import transcribe_features
from features.audio_processing import stream_48k_file_to_pipeline, stream_microphone_to_pipeline
from postprocess.onlineprocess import postprocess_online
import time


SAMPLE_RATE = 16_000
DECODE_WINDOW_SAMPLES = 48000  # 3 seconds
DECODE_STRIDE_SAMPLES = 4800 
FINALIZE_AFTER_SILENCE_MS = 700


def print_streaming_metrics(decode_latencies, rtfs):
    if not decode_latencies:
        return
    latencies_ms = np.array(decode_latencies) * 1000

    print("\nStreaming performance:")
    print(f"Samples: {len(decode_latencies)}")
    print(f"Mean latency: {latencies_ms.mean():.1f} ms")
    print(f"Median latency: {np.median(latencies_ms):.1f} ms")
    print(f"P95 latency: {np.percentile(latencies_ms, 95):.1f} ms")
    print(f"Mean RTF: {np.mean(rtfs):.3f}")
    
    
def decode_audio(
    audio: np.ndarray,
    mel_filterbank,
    model,
    decoder="greedy",
) -> str:
    
    """
    Takes an array of audio samples, extracts a log-Mel spectrogram for ASR processing,
    transcribes the spectrogram into text, applies online postprocessing, and returns
    the resulting string.

    Args:
        audio (np.ndarray): 1D array of audio samples.
        mel_filterbank (np.ndarray): Precomputed Mel filterbank used for feature extraction.
        model: ASR model used for inference.
        decoder (str): Decoding method, either "greedy" or "beam".

    Returns:
        str: Postprocessed decoded transcript.
    """         

    if len(audio) == 0:
        return ""

    features = waveform_to_log_mel(audio, mel_filterbank)

    prediction = transcribe_features(
        features,
        model,
        decoder=decoder)

    return postprocess_online(prediction)


def run_online(
    input_file: str | None,
    model,
    decoder="greedy",
):
    
    """
    Runs the online ASR pipeline using either live microphone input or simulated
    streaming from a prerecorded audio file. Incoming queue events are handled by
    type. Speech audio is accumulated in both a full-utterance buffer and a rolling
    48,000-sample window used for partial decoding. Once the rolling window is full,
    partial transcripts are updated every configured decode stride, replacing the
    previous partial transcript. Silence duration is accumulated separately, and
    after at least 700 ms of silence, the complete utterance is decoded and committed
    as a final transcript.
    
    Args:
        input_file (str | None): Path to a prerecorded audio file, or None for live
        microphone input.
        model: ASR model used for inference.
        decoder (str): Decoding method, either "greedy" or "beam".
        
    Returns:
        None
    
    """    

    pipeline_queue = queue.Queue(maxsize=100)

    mel_filterbank = build_mel_filterbank(
        sample_rate=SAMPLE_RATE,
        n_fft_bins=201,
        n_mels=80,
    )

    if input_file is None:
        producer_target = stream_microphone_to_pipeline
        producer_args = (pipeline_queue, )
    else:
        producer_target = stream_48k_file_to_pipeline
        producer_args = (
            input_file,
            pipeline_queue)
        
    decode_latencies = []
    rtfs = []    

    # Two different audio states

    # Contains ALL speech audio in the current utterance.
    utterance_chunks = []

    # Contains only recent audio for rolling partial decoding.
    rolling_buffer = np.empty(0, dtype=np.float32)

    # Controls how often partial decoding happens.
    samples_since_decode = 0

    accumulated_silence_ms = 0

    # Text state
    current_partial = ""
    committed_transcripts = []

    audio_thread = threading.Thread(
        target=producer_target,
        args=producer_args,
        daemon=True)

    audio_thread.start()

    print("Starting online ASR pipeline...")
    
    try:
        
        while True:
        
            event = pipeline_queue.get()
            event_type = event.get("type")
    
            # SPEECH
    
            if event_type == "speech":
    
                audio = event["audio"]
    
                # Save the audio permanently for this utterance.
                utterance_chunks.append(audio)
    
                # Also add it to the rolling partial-decoding buffer.
                rolling_buffer = np.concatenate(
                    (rolling_buffer, audio))
    
                samples_since_decode += len(audio)
    
                # Speech resumed.
                accumulated_silence_ms = 0
    
                # Keep at most the most recent 3 seconds
                # for partial inference.
                if len(rolling_buffer) > DECODE_WINDOW_SAMPLES:
                    rolling_buffer = rolling_buffer[-DECODE_WINDOW_SAMPLES:]
    
                # Partial decoding 
    
                if (
                    len(rolling_buffer) >= DECODE_WINDOW_SAMPLES
                    and
                    samples_since_decode >= DECODE_STRIDE_SAMPLES
                ):
                    
                    start = time.perf_counter()
    
                    partial_prediction = decode_audio(
                        rolling_buffer,
                        mel_filterbank,
                        model,
                        decoder=decoder)
                    
                    latency = time.perf_counter() - start
                    
                    audio_duration = len(rolling_buffer) / SAMPLE_RATE
                    rtf = latency / audio_duration
                    decode_latencies.append(latency)
                    rtfs.append(rtf)                
                    
    
                    if partial_prediction:
    
                        # Replace the previous partial.
                        current_partial = partial_prediction
    
                        print(
                            "\r\033[Kpartial transcript: "
                            + current_partial,
                            end="",
                            flush=True)                                      
    
                    samples_since_decode = 0
                    
            # SILENCE
    
            elif event_type == "silence":
    
                accumulated_silence_ms += event.get(
                    "duration_ms",
                    0)
    
                if (
                    accumulated_silence_ms
                    >= FINALIZE_AFTER_SILENCE_MS):
    
                    if utterance_chunks:
    
                        # Reconstruct the COMPLETE utterance.
                        utterance_audio = np.concatenate(
                            utterance_chunks)
    
                        # Final decode gets full utterance context.
                        final_prediction = decode_audio(
                            utterance_audio,
                            mel_filterbank,
                            model,
                            decoder=decoder)
    
                        if final_prediction:
    
                            committed_transcripts.append(final_prediction)
    
                            print("final transcript:",final_prediction)
                            print("committed transcript:"," ".join(committed_transcripts))
                       
                    # Reset for the next utterance
    
                    utterance_chunks = []
    
                    rolling_buffer = np.empty(
                        0,
                        dtype=np.float32)
    
                    current_partial = ""
                    samples_since_decode = 0
                    accumulated_silence_ms = 0
    
            # END OF STREAM
    
            elif event_type == "end":
                
                if utterance_chunks:
    
                    utterance_audio = np.concatenate(utterance_chunks)
    
                    final_prediction = decode_audio(
                        utterance_audio,
                        mel_filterbank,
                        model,
                        decoder=decoder)
    
                    if final_prediction:
    
                        committed_transcripts.append(
                            final_prediction)
                        
                        print()
                        print("final transcript:", final_prediction)
                        print("committed transcript:", " ".join(committed_transcripts))
    
                break
    
            else:
                print("Ignoring unknown queue event:", event)

    except KeyboardInterrupt:
        print("\nStopping online ASR...")
    
    else:
        # Only wait for producer on a normal stream ending.
        audio_thread.join()
    
    finally:
        print_streaming_metrics(
            decode_latencies,
            rtfs)
    
        print("Online ASR pipeline finished.")        