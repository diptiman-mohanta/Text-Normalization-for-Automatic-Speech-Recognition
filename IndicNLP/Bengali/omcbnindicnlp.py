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

# Try to import Indic NLP for Bengali normalization
try:
    from indicnlp import common
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    INDIC_RESOURCES_PATH = os.environ.get('INDIC_NLP_RESOURCES', './indic_nlp_resources')
    if os.path.exists(INDIC_RESOURCES_PATH):
        common.set_resources_path(INDIC_RESOURCES_PATH)
    INDIC_NLP_AVAILABLE = True
except ImportError:
    INDIC_NLP_AVAILABLE = False

# Bengali ASR Models
MODELS = {
    "arijit": "arijitx/wav2vec2-large-xlsr-bengali",
    "tanmoy": "tanmoyio/wav2vec2-large-xlsr-bengali",
    "shahruk": "shahruk10/wav2vec2-xls-r-300m-bengali-commonvoice",
    "ai4bharat": "ai4bharat/indicwav2vec_v1_bengali"
}

MODEL_NAME = MODELS["tanmoy"]  # Change model here

class BengaliTextNormalizer:
    def __init__(self):
        self.indic_normalizer = None
        if INDIC_NLP_AVAILABLE:
            try:
                self.indic_normalizer = IndicNormalizerFactory().get_normalizer("bn")
            except:
                print("Warning: Could not initialize Bengali normalizer")
    
    def normalize(self, text):
        if not text or text.strip() == "":
            return text
        
        normalized = text.strip()
        
        # Apply Unicode normalization (NFC form - canonical composition)
        normalized = unicodedata.normalize('NFC', normalized)
        
        # Apply Indic NLP normalization if available
        if self.indic_normalizer:
            try:
                normalized = self.indic_normalizer.normalize(normalized)
            except:
                pass
        
        # Remove punctuation: question marks, commas, and full stops
        # Including both English and Bengali punctuation marks
        punctuation_to_remove = [
            '?',      # Question mark
            ',',      # Comma
            '.',      # Full stop
            '।',      # Bengali full stop (dari)
            '!',      # Exclamation mark (optional)
        ]
        for punct in punctuation_to_remove:
            normalized = normalized.replace(punct, '')
        
        # Clean up whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized.strip()

def load_dataset(file_path="dataset_all2.json"):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {file_path}")
    return data

def load_model(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Try multiple loading strategies
    model = None
    loading_method = None
    
    # Strategy 1: Try safetensors first
    try:
        print("Attempting to load with safetensors...")
        model = Wav2Vec2ForCTC.from_pretrained(model_name, use_safetensors=True)
        loading_method = "safetensors"
        print("✓ Loaded successfully with safetensors")
    except Exception as e:
        print(f"Safetensors loading failed: {str(e)[:100]}")
    
    # Strategy 2: Try with trust_remote_code
    if model is None:
        try:
            print("Attempting to load with trust_remote_code...")
            model = Wav2Vec2ForCTC.from_pretrained(
                model_name, 
                trust_remote_code=True,
                use_safetensors=False
            )
            loading_method = "trust_remote_code"
            print("✓ Loaded successfully with trust_remote_code")
        except Exception as e:
            print(f"trust_remote_code loading failed: {str(e)[:100]}")
    
    # Strategy 3: Force PyTorch bin loading (requires PyTorch >= 2.6)
    if model is None:
        try:
            print("Attempting standard loading (requires PyTorch >= 2.6)...")
            model = Wav2Vec2ForCTC.from_pretrained(model_name)
            loading_method = "standard"
            print("✓ Loaded successfully with standard method")
        except Exception as e:
            print(f"\n{'='*70}")
            print("ERROR: Could not load model with any method!")
            print(f"{'='*70}")
            print(f"\nError details: {str(e)}")
            print("\nPossible solutions:")
            print("1. Upgrade PyTorch to version 2.6+:")
            print("   pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            print("\n2. Install safetensors:")
            print("   pip install safetensors")
            print("\n3. Try a different model (change MODEL_NAME in the script)")
            print(f"{'='*70}\n")
            raise
    
    print(f"Loading method used: {loading_method}")
    
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

def calculate_individual_metrics(ref, pred):
    """Calculate WER and CER for a single sample"""
    try:
        wer_score = wer(ref, pred) * 100
        cer_score = cer(ref, pred) * 100
    except:
        wer_score = 100.0
        cer_score = 100.0
    return wer_score, cer_score

def analyze_dataset(dataset, model, processor, device, sampling_rate, normalizer):
    results = []
    
    total_samples = len(dataset)
    print(f"Processing {total_samples} samples...")
    
    for i, sample in enumerate(dataset):
        try:
            sample_id = f"sample_{i+1:04d}"
            audio_path = sample["audio"]
            ref_text = sample.get("text", sample.get("transcript", ""))
            
            # Process audio
            start_time = time.time()
            audio_array = preprocess_audio(audio_path, sampling_rate)
            prediction_raw = transcribe_audio(audio_array, model, processor, device, sampling_rate)
            processing_time = time.time() - start_time
            
            # Normalize texts (includes unicode normalization + punctuation removal)
            ref_normalized = normalizer.normalize(ref_text)
            pred_normalized = normalizer.normalize(prediction_raw)
            
            # Calculate metrics for raw
            wer_raw, cer_raw = calculate_individual_metrics(ref_text, prediction_raw)
            
            # Calculate metrics for normalized
            wer_normalized, cer_normalized = calculate_individual_metrics(ref_normalized, pred_normalized)
            
            # Store results
            results.append({
                'sample_id': sample_id,
                'audio_path': audio_path,
                'reference': ref_text,
                'reference_normalized': ref_normalized,
                'prediction_raw': prediction_raw,
                'prediction_normalized': pred_normalized,
                'wer_raw': round(wer_raw, 2),
                'cer_raw': round(cer_raw, 2),
                'wer_normalized': round(wer_normalized, 2),
                'cer_normalized': round(cer_normalized, 2),
                'processing_time': round(processing_time, 3)
            })
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i+1}/{total_samples} samples")
        
        except Exception as e:
            print(f"Error processing sample {i+1}: {str(e)[:100]}")
            results.append({
                'sample_id': f"sample_{i+1:04d}",
                'audio_path': sample.get("audio", ""),
                'reference': sample.get("text", ""),
                'reference_normalized': "",
                'prediction_raw': "[ERROR]",
                'prediction_normalized': "[ERROR]",
                'wer_raw': 100.0,
                'cer_raw': 100.0,
                'wer_normalized': 100.0,
                'cer_normalized': 100.0,
                'processing_time': 0.0
            })
    
    return results

def calculate_overall_metrics(results):
    """Calculate overall statistics from individual results"""
    refs = [r['reference'] for r in results]
    refs_norm = [r['reference_normalized'] for r in results]
    preds = [r['prediction_raw'] for r in results]
    preds_norm = [r['prediction_normalized'] for r in results]
    
    overall_metrics = {
        'total_samples': len(results),
        'raw_wer': round(wer(refs, preds) * 100, 2),
        'raw_cer': round(cer(refs, preds) * 100, 2),
        'normalized_wer': round(wer(refs_norm, preds_norm) * 100, 2),
        'normalized_cer': round(cer(refs_norm, preds_norm) * 100, 2),
        'avg_wer_raw': round(sum(r['wer_raw'] for r in results) / len(results), 2),
        'avg_cer_raw': round(sum(r['cer_raw'] for r in results) / len(results), 2),
        'avg_wer_normalized': round(sum(r['wer_normalized'] for r in results) / len(results), 2),
        'avg_cer_normalized': round(sum(r['cer_normalized'] for r in results) / len(results), 2),
        'total_time': round(sum(r['processing_time'] for r in results), 2),
        'avg_time': round(sum(r['processing_time'] for r in results) / len(results), 3),
        'empty_predictions': sum(1 for r in results if r['prediction_raw'].strip() == ""),
    }
    
    overall_metrics['wer_improvement'] = round(overall_metrics['raw_wer'] - overall_metrics['normalized_wer'], 2)
    overall_metrics['cer_improvement'] = round(overall_metrics['raw_cer'] - overall_metrics['normalized_cer'], 2)
    
    return overall_metrics

def main():
    print("="*70)
    print("BENGALI ASR PERFORMANCE ANALYSIS")
    print("(with Unicode Normalization + Punctuation Removal)")
    print("="*70)
    
    # Load resources
    dataset = load_dataset()
    model, processor, device = load_model(MODEL_NAME)
    sampling_rate = processor.feature_extractor.sampling_rate
    normalizer = BengaliTextNormalizer()
    
    print(f"\nModel: {MODEL_NAME}")
    print(f"Indic NLP: {'Available' if INDIC_NLP_AVAILABLE else 'Not Available (using basic normalization)'}")
    print(f"Unicode Normalization: NFC (Canonical Composition)")
    print(f"Punctuation Removal: ?, , . । !")
    print()
    
    # Process dataset
    start_time = time.time()
    results = analyze_dataset(dataset, model, processor, device, sampling_rate, normalizer)
    total_time = time.time() - start_time
    
    # Calculate overall metrics
    overall_metrics = calculate_overall_metrics(results)
    
    # Print results
    print("\n" + "="*70)
    print("OVERALL RESULTS")
    print("="*70)
    print(f"\nRAW (no normalization):")
    print(f"  WER: {overall_metrics['raw_wer']:.2f}%")
    print(f"  CER: {overall_metrics['raw_cer']:.2f}%")
    
    print(f"\nNORMALIZED (Unicode + Punctuation Removal):")
    print(f"  WER: {overall_metrics['normalized_wer']:.2f}%")
    print(f"  CER: {overall_metrics['normalized_cer']:.2f}%")
    
    print(f"\nIMPROVEMENT:")
    print(f"  WER: {overall_metrics['wer_improvement']:+.2f} pp")
    print(f"  CER: {overall_metrics['cer_improvement']:+.2f} pp")
    
    print(f"\nPROCESSING:")
    print(f"  Samples: {overall_metrics['total_samples']}")
    print(f"  Total Time: {overall_metrics['total_time']:.2f}s")
    print(f"  Avg Time: {overall_metrics['avg_time']:.3f}s/sample")
    print(f"  Empty Predictions: {overall_metrics['empty_predictions']}")
    
    # Show sample results
    print("\n" + "="*70)
    print("SAMPLE RESULTS (first 3)")
    print("="*70)
    for i in range(min(3, len(results))):
        r = results[i]
        print(f"\n[{r['sample_id']}]")
        print(f"  REF:       {r['reference']}")
        print(f"  REF_NORM:  {r['reference_normalized']}")
        print(f"  PRED:      {r['prediction_raw']}")
        print(f"  PRED_NORM: {r['prediction_normalized']}")
        print(f"  WER: {r['wer_raw']:.1f}% → {r['wer_normalized']:.1f}%")
        print(f"  CER: {r['cer_raw']:.1f}% → {r['cer_normalized']:.1f}%")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('bengali_asr_results5.csv', index=False, encoding='utf-8')
    
    with open('bengali_asr_metrics5.json', 'w', encoding='utf-8') as f:
        json.dump(overall_metrics, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("✓ Results saved to:")
    print("  - bengali_asr_results5.csv")
    print("  - bengali_asr_metrics5.json")
    print("="*70)
    
    return results, overall_metrics

if __name__ == "__main__":
    results, metrics = main()
