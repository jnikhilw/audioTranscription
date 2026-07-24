from inference.decode import load_model
from pipeline.online import run_online
from pipeline.offline import run_offline

MODE = "online"
# model = load_model("checkpoints/best_val.pt")

# Set to None to activate your live 'stream_microphone_to_pipeline' thread!
# Set to a string (e.g., "audio.wav") to stream a pre-recorded file.
INPUT_SOURCE = None

def main():         
    if MODE == "online":
        print(f"[System] Booting Online ASR. Source: {'Live Microphone' if INPUT_SOURCE is None else INPUT_SOURCE}")
        model = load_model("checkpoints/best_val.pt")
        run_online(INPUT_SOURCE, model) 
        
    elif MODE == "offline":
        model = load_model("checkpoints/best_val.pt")
        run_offline(INPUT_SOURCE, model) 

        
        
if __name__ == "__main__":
    main()