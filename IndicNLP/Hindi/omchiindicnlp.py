################## Hindi Indic NLP########################
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import time
import json
import pandas as pd
from jiwer import wer, cer
import os
import re
import unicodedata

try:
    from indicnlp import common
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    INDIC_RESOURCES_PATH = os.environ.get('INDIC_NLP_RESOURCES', './indic_nlp_resources')
    if os.path.exists(INDIC_RESOURCES_PATH):
        common.set_resources_path(INDIC_RESOURCES_PATH)
    INDIC_NLP_AVAILABLE = True
except ImportError:
    INDIC_NLP_AVAILABLE = False

MODEL_NAME = "theainerd/Wav2Vec2-large-xlsr-hindi"
# MODEL_NAME = "ai4bharat/indicwav2vec-hindi"
# MODEL_NAME = "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200"
class HindiTextNormalizer:
    def __init__(self):
        self.indic_normalizer = None
        if INDIC_NLP_AVAILABLE:
            try:
                self.indic_normalizer = IndicNormalizerFactory().get_normalizer("hi")
            except:
                pass
    
    def normalize(self, text):
        if not text or text.strip() == "":
            return text
        
        normalized = text.strip()
        
        # Unicode normalization (NFC - Canonical Decomposition followed by Canonical Composition)
        normalized = unicodedata.normalize('NFC', normalized)
        
        # Apply Indic NLP normalization if available
        if self.indic_normalizer:
            try:
                normalized = self.indic_normalizer.normalize(normalized)
            except:
                pass
        
        # Remove punctuation: question mark, comma, full stop, and other common punctuation
        # Includes both ASCII and Unicode variants
        normalized = re.sub(r'[??,，。.!！:：;；\'\"''""]', '', normalized)
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip()

def load_dataset(file_path="dataset_all.json"):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    return {"test": data}

def load_model(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    
    # Use safetensors to avoid torch.load security vulnerability
    model = Wav2Vec2ForCTC.from_pretrained(model_name, use_safetensors=True)
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    return model, processor, device

def preprocess_audio(audio_path, sampling_rate):
    waveform, sr = torchaudio.load(audio_path)
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    if sr != sampling_rate:
        resampler = torchaudio.transforms.Resample(sr, sampling_rate)
        waveform = resampler(waveform)
    
    return waveform.squeeze().numpy()

def transcribe_audio(audio_array, model, processor, device, sampling_rate):
    inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    
    with torch.no_grad():
        logits = model(input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    return transcription
 # use this to transcribe while using vakyansh
"""
 def transcribe_audio(audio_array, model, processor, device, sampling_rate):
    inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    
    with torch.no_grad():
        logits = model(input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    
    # Fix: Add skip_special_tokens=True to remove <s> tokens
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    return transcription 
"""
def calculate_individual_metrics(reference, prediction):
    """Calculate WER and CER for a single sample"""
    try:
        if not reference.strip() or not prediction.strip():
            return 0.0, 0.0
        wer_score = wer(reference, prediction) * 100
        cer_score = cer(reference, prediction) * 100
        return wer_score, cer_score
    except:
        return 0.0, 0.0

def analyze_dataset(dataset_split, model, processor, device, sampling_rate, normalizer):
    sample_ids = []
    audio_paths = []
    references = []
    references_normalized = []
    predictions_raw = []
    predictions_normalized = []
    processing_times = []
    raw_wers = []
    raw_cers = []
    normalized_wers = []
    normalized_cers = []
    
    total_samples = len(dataset_split)
    print(f"Processing {total_samples} samples...")
    
    for i, sample in enumerate(dataset_split):
        try:
            # Extract sample_id and audio_path
            sample_id = sample.get("id", f"sample_{i+1}")
            audio_path = sample["audio"]
            
            sample_ids.append(sample_id)
            audio_paths.append(audio_path)
            
            ref_text = sample.get("text", sample.get("transcript", ""))
            references.append(ref_text)
            ref_normalized = normalizer.normalize(ref_text)
            references_normalized.append(ref_normalized)
            
            start_time = time.time()
            audio_array = preprocess_audio(audio_path, sampling_rate)
            prediction_raw = transcribe_audio(audio_array, model, processor, device, sampling_rate)
            predictions_raw.append(prediction_raw)
            
            prediction_normalized = normalizer.normalize(prediction_raw)
            predictions_normalized.append(prediction_normalized)
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Calculate individual metrics for this sample
            raw_wer, raw_cer = calculate_individual_metrics(ref_text, prediction_raw)
            raw_wers.append(raw_wer)
            raw_cers.append(raw_cer)
            
            norm_wer, norm_cer = calculate_individual_metrics(ref_normalized, prediction_normalized)
            normalized_wers.append(norm_wer)
            normalized_cers.append(norm_cer)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{total_samples} samples")
        
        except Exception as e:
            print(f"Error processing sample {i+1}: {str(e)[:100]}")
            sample_ids.append(f"sample_{i+1}")
            audio_paths.append("")
            references.append("")
            references_normalized.append("")
            predictions_raw.append("")
            predictions_normalized.append("")
            processing_times.append(0)
            raw_wers.append(0.0)
            raw_cers.append(0.0)
            normalized_wers.append(0.0)
            normalized_cers.append(0.0)
    
    return (sample_ids, audio_paths, references, references_normalized, predictions_raw, 
            predictions_normalized, processing_times, raw_wers, raw_cers, normalized_wers, normalized_cers)

def calculate_metrics(references, predictions, label):
    wer_score = wer(references, predictions)
    cer_score = cer(references, predictions)
    
    metrics = {
        f"{label}_wer": wer_score * 100,
        f"{label}_cer": cer_score * 100,
        f"{label}_empty": sum(1 for pred in predictions if pred.strip() == ""),
    }
    
    return metrics

def main():
    print("="*70)
    print("HINDI ASR PERFORMANCE ANALYSIS")
    print("="*70)
    
    dataset = load_dataset()
    model, processor, device = load_model(MODEL_NAME)
    sampling_rate = processor.feature_extractor.sampling_rate
    normalizer = HindiTextNormalizer()
    
    print(f"Indic NLP: {'Available' if INDIC_NLP_AVAILABLE else 'Not Available'}")
    print()
    
    start_time = time.time()
    (sample_ids, audio_paths, references, references_normalized, predictions_raw, 
     predictions_normalized, processing_times, raw_wers, raw_cers, normalized_wers, normalized_cers) = analyze_dataset(
        dataset["test"], model, processor, device, sampling_rate, normalizer
    )
    total_time = time.time() - start_time
    
    raw_metrics = calculate_metrics(references, predictions_raw, "raw")
    normalized_metrics = calculate_metrics(references_normalized, predictions_normalized, "normalized")
    
    combined_metrics = {
        **raw_metrics, 
        **normalized_metrics,
        "total_samples": len(references),
        "total_time": total_time,
        "avg_time": total_time / len(references)
    }
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nRAW (no normalization):")
    print(f"  WER: {raw_metrics['raw_wer']:.2f}%")
    print(f"  CER: {raw_metrics['raw_cer']:.2f}%")
    print(f"  Empty: {raw_metrics['raw_empty']}")
    
    print(f"\nNORMALIZED (with punctuation removal + Unicode + Indic NLP):")
    print(f"  WER: {normalized_metrics['normalized_wer']:.2f}%")
    print(f"  CER: {normalized_metrics['normalized_cer']:.2f}%")
    print(f"  Empty: {normalized_metrics['normalized_empty']}")
    
    wer_diff = raw_metrics['raw_wer'] - normalized_metrics['normalized_wer']
    cer_diff = raw_metrics['raw_cer'] - normalized_metrics['normalized_cer']
    
    print(f"\nIMPROVEMENT:")
    print(f"  WER: {wer_diff:+.2f} pp {'(better)' if wer_diff > 0 else '(worse)'}")
    print(f"  CER: {cer_diff:+.2f} pp {'(better)' if cer_diff > 0 else '(worse)'}")
    
    print(f"\nPROCESSING:")
    print(f"  Samples: {combined_metrics['total_samples']}")
    print(f"  Total: {total_time:.2f}s")
    print(f"  Avg: {combined_metrics['avg_time']:.3f}s/sample")
    
    print("\n" + "="*70)
    print("SAMPLE COMPARISONS (first 5)")
    print("="*70)
    for i in range(min(5, len(references))):
        print(f"\n[{i+1}] REF: {references[i]}")
        print(f"    REF_NORM: {references_normalized[i]}")
        print(f"    PRED: {predictions_raw[i]}")
        print(f"    PRED_NORM: {predictions_normalized[i]}")
    
    results_df = pd.DataFrame({
        'sample_id': sample_ids,
        'audio_path': audio_paths,
        'reference': references,
        'reference_normalized': references_normalized,
        'prediction_raw': predictions_raw,
        'prediction_normalized': predictions_normalized,
        'wer_raw': raw_wers,
        'cer_raw': raw_cers,
        'wer_normalized': normalized_wers,
        'cer_normalized': normalized_cers,
        'processing_time': processing_times
    })
    results_df.to_csv('asr_results2.csv', index=False, encoding='utf-8')
    
    with open('asr_metrics2.json', 'w', encoding='utf-8') as f:
        json.dump(combined_metrics, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("Saved: asr_results2.csv, asr_metrics2.json")
    print("="*70)
    
    return combined_metrics

if __name__ == "__main__":
    metrics = main()
