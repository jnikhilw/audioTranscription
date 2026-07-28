from inference.decode import load_model, transcribe_features
import random
from dataset.asr_vocab import VOCAB
from dataset.asr_dataset import LibriSpeechASRDataset
from training.utils import get_librispeech_split_range
import torch 

# train-set 
start_percent_train = 0.0
end_percent_train = 0.95

# val-set 
start_percent_val = 0.95
end_percent_val = .97


start_train, limit_train =  get_librispeech_split_range(root="data/librispeech",
    url="train-other-500",
    start_percent= start_percent_train,
    end_percent= end_percent_train)


start_val, limit_val =  get_librispeech_split_range(root="data/librispeech",
    url="train-other-500",
    start_percent= start_percent_val,
    end_percent= end_percent_val)


eval_dataset_train = LibriSpeechASRDataset(
    root="data/librispeech",
    url="train-other-500",
    start=start_train,
    limit=limit_train,
    use_cache=True,
    return_transcript=True

)

eval_dataset_val = LibriSpeechASRDataset(
    root="data/librispeech",
    url="train-other-500",
    start=start_val,
    limit=limit_val,
    use_cache=True,
    return_transcript=True

)

def edit_distance(a: str, b: str) -> int:

    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        dp[i][0] = i

    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):

        for j in range(1, len(b) + 1):

            if a[i - 1] == b[j - 1]:
                cost = 0

            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    return dp[len(a)][len(b)]


def character_error_rate(truth: str, pred: str) -> float:
    
    if len(truth) == 0:
        return 0.0 if len(pred) == 0 else 1.0

    return edit_distance(truth, pred) / len(truth)


VOCAB_CHARS = [token for token in VOCAB if token != "<blank>"]


def random_prediction(length: int) -> str:
    return "".join(random.choice(VOCAB_CHARS) for _ in range(length))


def eval_diagnostics(dataset, inspect_predictions = True, skill_score = True, DECODER = "greedy"):
    model = load_model("checkpoints/train_other_500_latest.pt")
    total_cer = 0
    
    for i in range(len(dataset)):
    
        features, target_ids, truth = dataset[i]  
        pred = transcribe_features(features,model, decoder= DECODER)
        total_cer += character_error_rate(truth, pred)
        
        if inspect_predictions:
            print("truth:", truth)
            print("pred :", pred)
            print("CER  :", character_error_rate(truth, pred))
        
    avg_cer =  total_cer / len(dataset)
    
    print("average CER:", avg_cer)
    
    if not inspect_predictions and not skill_score:
        return avg_cer
    
    if skill_score:
        
        trials = 10
        trial_cers = []
        
        for _ in range(trials):
      
            total_cer = 0.0
        
            for i in range(len(dataset)):
                
                features, target_ids, truth = dataset[i]
                pred = random_prediction(len(truth))
                total_cer += character_error_rate(truth, pred)
        
            trial_cers.append(total_cer / len(dataset))
        
        baseline_cer = sum(trial_cers) / len(trial_cers)
        
        print("random baseline CER mean:", baseline_cer)   
        print("average CER:", avg_cer)        
        print("Skill score:", 1 - (avg_cer/baseline_cer))


        
print(eval_diagnostics(eval_dataset_val, inspect_predictions = True, skill_score = True, DECODER = "greedy"))

