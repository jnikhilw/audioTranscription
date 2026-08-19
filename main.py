from inference.decode import load_model
from pipeline.online import run_online


MODE = "online"
# Set to None to activate live microphone streaming.
# Set to a file path (e.g., "audio.wav") to stream pre-recorded audio.
INPUT_SOURCE = None


def main():         
    if MODE == "online":
        print(f"[System] Booting Online ASR. Source: {'Live Microphone' if INPUT_SOURCE is None else INPUT_SOURCE}")
        model = load_model("checkpoints/train_other_500_latest.pt")
        run_online(INPUT_SOURCE, model) 
     
    elif MODE == "offline":
        raise NotImplementedError("Offline mode is not currently available.")
 
        
if __name__ == "__main__":
    main()