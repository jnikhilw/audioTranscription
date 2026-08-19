import time
import wave
import numpy as np
import scipy.signal as signal
from pyaudio import PyAudio, paInt16
import scipy.io.wavfile as wavfile
import threading
import queue

# We remove the global RNNoise import at the top!
_global_denoiser = None
_rnnoise_failed = False

def get_rnnoise_instance():
    global _global_denoiser
    if _global_denoiser is None:
        # Lazy import: This only runs when the denoiser is actually called
        from pyrnnoise import RNNoise 
        _global_denoiser = RNNoise(sample_rate=48000)
    return _global_denoiser

def rnnoise_denoise_48k_chunk(chunk_48k_int16):
    """
    CORE FUNCTION: RNNoise only. No VAD dropping. No downsampling.
    Returns: (clean_48k_float, avg_speech_prob)
    """
    denoiser = get_rnnoise_instance()
    
    clean_frames = []
    speech_probabilities = []
    
    for speech_prob, denoised_frame in denoiser.denoise_chunk(chunk_48k_int16):
        clean_frames.append(denoised_frame.flatten()) 
        speech_probabilities.append(np.mean(speech_prob))
        
    if not clean_frames:
        return np.array([], dtype=np.float32), 0.0
        
    avg_speech_prob = sum(speech_probabilities) / len(speech_probabilities)
    clean_48k_float = np.concatenate(clean_frames)
    
    return clean_48k_float.astype(np.float32), avg_speech_prob

def _apply_dsp_filters_16k(clean_48k_float: np.ndarray) -> np.ndarray: 
    """
    HELPER FUNCTION: Keeps our math DRY. Applies downsampling, pre-emphasis, and DC removal.
    """
    clean_16k_float = signal.resample_poly(clean_48k_float, up=1, down=3).astype(np.float32)
    # clean_16k_float = np.append(clean_16k_float[0], clean_16k_float[1:] - 0.97 * clean_16k_float[:-1]) # pre emphasis filter
    # clean_16k_float = clean_16k_float - np.mean(clean_16k_float) # DC offset removal.
    clean_16k_float /= 32768.0
    return clean_16k_float

def kyle_online_preprocess_chunk(chunk_48k_int16, vad_threshold=0.2, chunk_duration_ms=100, enable_rnnoise=True): 
    """
    LIVE APP FUNCTION: Calls RNNoise core (if enabled). 
    Returns exactly the dictionary format Nikhil requested.
    """
    global _rnnoise_failed
    
    avg_speech_prob = 0.0
    clean_48k_float = np.array([], dtype=np.float32)
    
    # 1. Attempt to use the AI Denoiser
    if enable_rnnoise and not _rnnoise_failed:
        try:
            clean_48k_float, avg_speech_prob = rnnoise_denoise_48k_chunk(chunk_48k_int16)
        except Exception as e:
            print(f"\n[System Warning] RNNoise rejected by Windows ({e}). Automatically routing to Energy VAD.")
            _rnnoise_failed = True # Flips the switch so it doesn't spam the terminal every 100ms
            
    # 2. Fallback if disabled OR if Windows crashed the import
    if not enable_rnnoise or _rnnoise_failed:
        clean_48k_float = chunk_48k_int16.astype(np.float32)
        rms_energy = np.sqrt(np.mean(clean_48k_float**2))
        avg_speech_prob = 1.0 if rms_energy > 300.0 else 0.0 
    
    # If silence or empty, return the silence heartbeat dictionary
    if avg_speech_prob < vad_threshold or len(clean_48k_float) == 0:
        return {"type": "silence", "duration_ms": chunk_duration_ms}
        
    # If speech, apply DSP and return the speech dictionary
    clean_16k_chunk = _apply_dsp_filters_16k(clean_48k_float)
    return {"type": "speech", "audio": clean_16k_chunk}

def kyle_training_preprocess_chunk(chunk_48k_int16, enable_rnnoise=True):
    """
    TRAINING FUNCTION: Calls RNNoise core. NEVER drops audio. Downsamples everything.
    """
    global _rnnoise_failed
    
    clean_48k_float = np.array([], dtype=np.float32)
    
    if enable_rnnoise and not _rnnoise_failed:
        try:
            clean_48k_float, _ = rnnoise_denoise_48k_chunk(chunk_48k_int16)
        except Exception:
            _rnnoise_failed = True
            
    if not enable_rnnoise or _rnnoise_failed:
        clean_48k_float = chunk_48k_int16.astype(np.float32)
        
    if len(clean_48k_float) == 0:
        return np.array([], dtype=np.float32)
        
    return _apply_dsp_filters_16k(clean_48k_float)


# used for reading wav files instead of using live microphone
def stream_48k_file_to_pipeline(file_path, pipeline_queue, chunk_size=4800, enable_rnnoise=True): 
    try:
        with wave.open(file_path, 'rb') as wf:
            assert wf.getframerate() == 48000, "Test file must be 48kHz!"
            assert wf.getnchannels() == 1, "Test file must be Mono, not Stereo!"
            
            chunk_duration = chunk_size / 48000.0 

            while True:
                raw_bytes = wf.readframes(chunk_size)
                
                if not raw_bytes:
                    break
                    
                if len(raw_bytes) < chunk_size * 2: 
                    raw_bytes = raw_bytes.ljust(chunk_size * 2, b'\x00')
                    
                chunk_48k_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
                
                payload = kyle_online_preprocess_chunk(
                    chunk_48k_int16, 
                    vad_threshold=0.3, 
                    chunk_duration_ms=100,
                    enable_rnnoise=enable_rnnoise
                )
                pipeline_queue.put(payload, block=True)
                
                time.sleep(chunk_duration)
    except Exception as e:
        print(f"Background thread crashed: {e}")
    finally:
        pipeline_queue.put({"type": "end"})
        

def stream_microphone_to_pipeline(pipeline_queue, chunk_size=4800, enable_rnnoise=True): 
    """
    LIVE MICROPHONE PRODUCER THREAD:
    Captures live 48kHz audio from the microphone and pushes it to the queue.
    """
    p = PyAudio()
    
    try:
        stream = p.open(
            format=paInt16,
            channels=1,
            rate=48000,
            input=True,
            frames_per_buffer=chunk_size
        )
        
        print(f"Microphone is live (RNNoise Requested: {'ON' if enable_rnnoise else 'OFF'}). Start speaking... (Press Ctrl+C to stop)")
        
        while True:
            raw_bytes = stream.read(chunk_size, exception_on_overflow=False)
            chunk_48k_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
            
            payload = kyle_online_preprocess_chunk(
                chunk_48k_int16, 
                vad_threshold=0.3, 
                chunk_duration_ms=100,
                enable_rnnoise=enable_rnnoise
            )
            pipeline_queue.put(payload, block=True)
            
    except Exception as e:
        print(f"Microphone thread crashed: {e}")
    finally:
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()
        pipeline_queue.put({"type": "end"})

if __name__ == "__main__":
    print("Starting Audio Processing Test")
    
    test_queue = queue.Queue(maxsize=100) 
    output_file = "clean_output_16k.wav"
    
    stream_thread = threading.Thread(
        target=stream_microphone_to_pipeline, 
        args=(test_queue,),
        kwargs={"enable_rnnoise": True}, 
        daemon=True
    )
    stream_thread.start()
    
    collected_chunks = []
    print("Listening to queue and collecting clean audio...")
    
    try:
        while True:
            payload = test_queue.get(timeout=5.0)
            
            if payload["type"] == "end":
                print("End signal received. Completely finished.")
                break
                
            if payload["type"] == "speech":
                collected_chunks.append(payload["audio"]) 
            elif payload["type"] == "silence":
                pass
                
            test_queue.task_done()
    except KeyboardInterrupt:
        print("\nStopping capture via keyboard interrupt")
            
    print("File streaming finished. Stitching audio back together")
    
    if collected_chunks:
        full_audio_float32 = np.concatenate(collected_chunks)
        max_amp = np.max(np.abs(full_audio_float32))
        print(f"Diagnostic: Max audio amplitude is {max_amp:.2f}")
        
        if max_amp > 32767.0:
            print("Audio clipped! Hard limiting boundaries.")
            safe_audio = np.clip(full_audio_float32, -32768.0, 32767.0)
        else:
            if max_amp > 0:
                safe_audio = (full_audio_float32 / max_amp) * 0.9 * 32767.0
            else:
                safe_audio = full_audio_float32
                
        full_audio_int16 = safe_audio.astype(np.int16)
        wavfile.write(output_file, 16000, full_audio_int16)
        print(f"Success. isolated vocals here: {output_file}")
    else:
        print("No audio was collected.")