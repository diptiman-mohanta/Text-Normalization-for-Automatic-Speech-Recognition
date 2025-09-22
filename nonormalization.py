import torch
import torchaudio
from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2ForCTC, Wav2Vec2Processor
import time
import json
import pandas as pd
from jiwer import wer, cer
import numpy as np

MODEL_OPTIONS = {
    "ai4bharat": "ai4bharat/indicwav2vec-hindi",
    "theainerd": "theainerd/Wav2Vec2-large-xlsr-hindi", 
    "vakyansh": "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200",
    "multilingual": "facebook/wav2vec2-large-xlsr-53"
}

selected_model = "ai4bharat"
model_name = MODEL_OPTIONS[selected_model]

def load_json_dataset(file_path):
    """Load dataset from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_custom_dataset():
    """Load dataset from dataset_all.json file"""
    file_path = "dataset_all.json"
    data = load_json_dataset(file_path)
    
    if data:
        print(f"Loaded dataset_all.json: {len(data)} samples")
        return {"test": data}
    else:
        print("dataset_all.json not found")
        return None

def load_model_safely(model_name):
    """Load model with fallback strategies"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    strategies = [
        ("AutoModel", lambda: (AutoModelForCTC.from_pretrained(model_name), AutoProcessor.from_pretrained(model_name))),
        ("Wav2Vec2", lambda: (Wav2Vec2ForCTC.from_pretrained(model_name), Wav2Vec2Processor.from_pretrained(model_name))),
        ("Fallback", lambda: (AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h"), AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")))
    ]
    
    for strategy_name, load_func in strategies:
        try:
            print(f"Trying {strategy_name} strategy...")
            model, processor = load_func()
            model = model.to(device)
            print(f"Successfully loaded model with {strategy_name}")
            return model, processor, device
        except Exception as e:
            print(f"{strategy_name} failed: {str(e)[:100]}...")
            continue
    
    raise Exception("All model loading strategies failed")

def preprocess_audio(audio_path):
    """Preprocess audio file"""
    try:
        waveform, sr = torchaudio.load(audio_path)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if sr != sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, sampling_rate)
            waveform = resampler(waveform)
        
        return waveform.squeeze().numpy()
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return torch.randn(sampling_rate).numpy()

def transcribe_audio(audio_array):
    """Transcribe audio array to text"""
    try:
        inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        
        with torch.no_grad():
            logits = model(input_values).logits.cpu()
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        
        return transcription
    except Exception as e:
        print(f"Transcription error: {e}")
        return "[ERROR]"

def analyze_dataset(dataset_split):
    """Analyze dataset and return results"""
    references = []
    predictions = []
    processing_times = []
    
    total_samples = len(dataset_split)
    print(f"Processing {total_samples} samples...")
    
    for i, sample in enumerate(dataset_split):
        ref_text = sample.get("text", sample.get("transcript", "Unknown"))
        references.append(ref_text)
        
        start_time = time.time()
        audio_path = sample["audio"]
        audio_array = preprocess_audio(audio_path)
        prediction = transcribe_audio(audio_array)
        predictions.append(prediction)
        
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{total_samples} samples")
    
    return references, predictions, processing_times

def calculate_metrics(references, predictions):
    """Calculate WER and CER metrics"""
    try:
        wer_score = wer(references, predictions)
        cer_score = cer(references, predictions)
        
        metrics = {
            "wer_percentage": wer_score * 100,
            "cer_percentage": cer_score * 100,
            "total_samples": len(references),
            "empty_predictions": sum(1 for pred in predictions if pred.strip() == ""),
        }
        
        metrics["empty_prediction_rate"] = (metrics["empty_predictions"] / metrics["total_samples"]) * 100
        return metrics
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {
            "wer_percentage": -1,
            "cer_percentage": -1,
            "total_samples": len(references),
            "empty_predictions": -1,
            "empty_prediction_rate": -1
        }

def main():
    """Main execution function"""
    print("Hindi ASR Performance Analysis")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    
    start_time = time.time()
    references, predictions, processing_times = analyze_dataset(dataset["test"])
    total_time = time.time() - start_time
    
    metrics = calculate_metrics(references, predictions)
    
    print("\nResults:")
    print(f"Word Error Rate (WER): {metrics['wer_percentage']:.2f}%")
    print(f"Character Error Rate (CER): {metrics['cer_percentage']:.2f}%")
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Empty Predictions: {metrics['empty_predictions']}")
    print(f"Total Processing Time: {total_time:.2f} seconds")
    print(f"Average Time per Sample: {total_time/len(references):.3f} seconds")
    
    print("\nSample Results:")
    for i in range(min(3, len(references))):
        print(f"Sample {i+1}:")
        print(f"Reference:  {references[i]}")
        print(f"Prediction: {predictions[i]}")
        print()
    
    try:
        results_df = pd.DataFrame({
            'reference': references,
            'prediction': predictions,
            'processing_time': processing_times
        })
        results_df.to_csv('hindi_asr_results.csv', index=False, encoding='utf-8')
        
        with open('hindi_asr_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print("Results saved to:")
        print("- hindi_asr_results.csv")
        print("- hindi_asr_metrics.json")
        
    except Exception as e:
        print(f"Could not save results: {e}")
    
    return references, predictions, metrics

if __name__ == "__main__":
    try:
        dataset = load_custom_dataset()
        if not dataset:
            print("Failed to load dataset")
            exit(1)
            
        model, processor, device = load_model_safely(model_name)
        sampling_rate = processor.feature_extractor.sampling_rate
        
        references, predictions, metrics = main()
        print(f"\nAnalysis completed. Final WER: {metrics['wer_percentage']:.2f}%")
        
    except Exception as e:
        print(f"Analysis failed: {e}")