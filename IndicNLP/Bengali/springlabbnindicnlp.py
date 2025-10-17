import json
import torch
import fairseq
import soundfile
import torch.nn.functional as F
from pathlib import Path
from jiwer import wer, cer
from typing import List, Dict
import re
import numpy as np
import unicodedata
import csv

# Indic NLP Library imports
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize


class BengaliASREvaluator:
    def __init__(self, model_path: str, use_cpu: bool = False):
        """
        Initialize the ASR evaluator with model path.
        Uses fairseq's checkpoint loading method for SPRING-INX models.
        
        Args:
            model_path: Path to SPRING_INX_wav2vec2_Bengali.pt
            use_cpu: Force CPU usage even if GPU is available
        """
        # Force CPU if requested or if CUDA is not available
        if use_cpu or not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda')
        
        print(f"Using device: {self.device}")
        
        # Clear GPU cache if using CUDA
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Load model using fairseq
        print("Loading model checkpoint (this may take a moment)...")
        self.model, self.cfg, self.task = self.load_model(model_path)
        
        # Get dictionary from task
        self.dictionary = self.task.target_dictionary
        print(f"Loaded dictionary with {len(self.dictionary)} tokens")
        print("Model loaded successfully")
        
        # Initialize Indic NLP normalizer for Bengali
        print("Initializing Indic NLP normalizer for Bengali...")
        self.indic_normalizer = IndicNormalizerFactory().get_normalizer("bn")
        print("Indic NLP normalizer initialized")
        
    def load_model(self, model_path: str):
        """Load the fairseq model using checkpoint_utils."""
        # Load model ensemble and task using fairseq
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [model_path],
            arg_overrides={"data": "/"}  # Dummy data path
        )
        
        model = models[0]
        model.to(self.device)
        model.eval()
        
        # Use FP16 for GPU to save memory
        if self.device.type == 'cuda':
            model.half()
            torch.cuda.empty_cache()
        
        return model, cfg, task
    
    def load_and_preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load and preprocess audio file (Windows compatible version).
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed audio tensor
        """
        # Read audio file
        audio, rate = soundfile.read(audio_path, dtype="float32")
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio)
        
        # If stereo, convert to mono
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.mean(dim=-1)
        
        # Apply gain normalization (equivalent to sox "gain -n")
        # This normalizes the audio to have peak amplitude of 1.0
        max_amplitude = torch.abs(audio_tensor).max()
        if max_amplitude > 0:
            audio_tensor = audio_tensor / max_amplitude
        
        # Add batch dimension
        input_sample = audio_tensor.unsqueeze(0)
        
        # Convert to appropriate dtype and device
        if self.device.type == 'cuda':
            input_sample = input_sample.half().to(self.device)
        else:
            input_sample = input_sample.float().to(self.device)
        
        # Apply layer normalization
        with torch.no_grad():
            input_sample = F.layer_norm(input_sample, input_sample.shape)
        
        return input_sample
    
    def is_bengali_char(self, char: str) -> bool:
        """Check if character is in Bengali Unicode range."""
        if not char:
            return False
        return '\u0980' <= char[0] <= '\u09FF'
    
    def join_bengali_characters(self, text: str) -> str:
        """
        Join separated Bengali characters back into complete words.
        Uses | as primary word boundary, with fallback heuristics.
        
        Args:
            text: Space-separated Bengali characters with | as word boundaries
            
        Returns:
            Properly joined Bengali text
        """
        # Split by spaces to get individual characters
        chars = text.split()
        
        if not chars:
            return ""
        
        result = []
        current_word = []
        
        # Dependent vowel signs (matras) that must attach to previous character
        dependent_vowels = set(['া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ'])
        
        # Other combining marks
        combining_marks = set(['ং', 'ঃ', 'ঁ', '্'])
        
        # Punctuation
        punctuation = set(['।', ',', '?', '!', '.', ':', ';', ')', '('])
        
        for i, char in enumerate(chars):
            # Handle word boundary marker
            if char == '|':
                if current_word:
                    result.append(''.join(current_word))
                    current_word = []
                continue
            
            # Handle punctuation
            if char in punctuation:
                if current_word:
                    result.append(''.join(current_word))
                    current_word = []
                result.append(char)
                continue
            
            # Handle Bengali characters
            if self.is_bengali_char(char):
                # Always join dependent vowels and combining marks
                if char in dependent_vowels or char in combining_marks:
                    current_word.append(char)
                # If previous character was hasant (্), join this character
                elif current_word and current_word[-1] == '্':
                    current_word.append(char)
                # Otherwise, add to current word (we'll rely on | for word boundaries)
                else:
                    current_word.append(char)
            else:
                # Non-Bengali character, treat as word boundary
                if current_word:
                    result.append(''.join(current_word))
                    current_word = []
                if char.strip():
                    result.append(char)
        
        # Add any remaining word
        if current_word:
            result.append(''.join(current_word))
        
        # Join with spaces
        final_text = ' '.join(result)
        
        # Clean up multiple spaces
        final_text = re.sub(r'\s+', ' ', final_text)
        
        # Remove space before punctuation
        final_text = re.sub(r'\s+([।,?!.;:])', r'\1', final_text)
        
        # Normalize Unicode
        final_text = unicodedata.normalize('NFC', final_text)
        
        return final_text.strip()
    
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio file to text using the loaded model.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            # Load and preprocess audio
            input_sample = self.load_and_preprocess_audio(audio_path)
            
            # Run inference
            with torch.no_grad():
                logits = self.model(source=input_sample, padding_mask=None)["encoder_out"]
            
            # Get predictions
            predicted_ids = torch.argmax(logits[:, 0], axis=-1)
            
            # Remove consecutive duplicates (CTC blank removal)
            predicted_ids = torch.unique_consecutive(predicted_ids).tolist()
            
            # Decode using dictionary
            transcription = self.dictionary.string(predicted_ids)
            
            # Debug: print raw transcription to understand the format
            print(f"RAW OUTPUT: {repr(transcription)}")
            
            # Join Bengali characters properly
            transcription = self.join_bengali_characters(transcription)
            
            # Final cleanup
            transcription = re.sub(r'\s+', ' ', transcription).strip()
            
            return transcription
            
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def normalize_text_basic(self, text: str) -> str:
        """Basic normalization for Bengali text (used for raw metrics)."""
        # Normalize Unicode (NFC normalization for Bengali)
        text = unicodedata.normalize('NFC', text)
        
        # Remove punctuation (question mark, comma, full stop, and Bengali danda)
        text = re.sub(r'[?,।.!;:()"\'\[\]{}]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def normalize_text_indic(self, text: str) -> str:
        """
        Normalize Bengali text using Indic NLP Library with additional normalizations.
        This performs comprehensive normalization for fair comparison.
        
        Args:
            text: Input Bengali text
            
        Returns:
            Normalized Bengali text
        """
        # First apply Indic NLP normalization
        text = self.indic_normalizer.normalize(text)
        
        # Normalize Unicode (NFC normalization)
        text = unicodedata.normalize('NFC', text)
        
        # Remove punctuation (question mark, comma, full stop, and Bengali danda)
        text = re.sub(r'[?,।.!;:()"\'\[\]{}]', '', text)
        
        # Additional Bengali-specific normalizations
        # Normalize য-ফলা (য়) and য variations
        text = text.replace('য়', 'য')
        
        # Normalize ref/reph variations
        text = text.replace('ৰ', 'র')
        
        # Normalize common character variants
        text = text.replace('ৎ', 'ত')
        
        # Normalize ড় to ড and ঢ় to ঢ (nukta variants)
        text = text.replace('ড়', 'ড')
        text = text.replace('ঢ়', 'ঢ')
        
        # Normalize ং to ঙ where appropriate
        # (keeping ং as is since it's used differently than ঙ)
        
        # Remove extra whitespace and normalize spacing
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Convert to lowercase (if applicable for Bengali digits/English text)
        text = text.lower()
        
        return text
    
    def evaluate_dataset(self, json_path: str, output_txt: str = "evaluation_results.txt", output_csv: str = "evaluation_results.csv"):
        """
        Evaluate the entire dataset and calculate both raw and normalized WER and CER.
        
        Args:
            json_path: Path to JSON file containing audio paths and reference texts
            output_txt: Path to save detailed text results
            output_csv: Path to save results in CSV format
        """
        # Load dataset
        with open(json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print(f"\nEvaluating {len(dataset)} audio files...")
        
        # Separate lists for raw and normalized metrics
        raw_references = []
        raw_hypotheses = []
        normalized_references = []
        normalized_hypotheses = []
        results = []
        
        for idx, item in enumerate(dataset):
            audio_path = item['audio']
            reference_original = item['text']
            
            print(f"\nProcessing {idx + 1}/{len(dataset)}: {Path(audio_path).name}")
            
            # Transcribe
            hypothesis_original = self.transcribe(audio_path)
            
            # Apply basic normalization for raw metrics (only Unicode normalization)
            reference_raw = self.normalize_text_basic(reference_original)
            hypothesis_raw = self.normalize_text_basic(hypothesis_original)
            
            # Apply Indic NLP normalization for normalized metrics
            reference_normalized = self.normalize_text_indic(reference_original)
            hypothesis_normalized = self.normalize_text_indic(hypothesis_original)
            
            # Store for overall metrics calculation
            raw_references.append(reference_raw)
            raw_hypotheses.append(hypothesis_raw)
            normalized_references.append(reference_normalized)
            normalized_hypotheses.append(hypothesis_normalized)
            
            # Calculate raw metrics for this sample
            sample_raw_wer = wer(reference_raw, hypothesis_raw) * 100 if reference_raw and hypothesis_raw else 100.0
            sample_raw_cer = cer(reference_raw, hypothesis_raw) * 100 if reference_raw and hypothesis_raw else 100.0
            
            # Calculate normalized metrics for this sample
            sample_norm_wer = wer(reference_normalized, hypothesis_normalized) * 100 if reference_normalized and hypothesis_normalized else 100.0
            sample_norm_cer = cer(reference_normalized, hypothesis_normalized) * 100 if reference_normalized and hypothesis_normalized else 100.0
            
            result = {
                'audio': audio_path,
                'reference_original': reference_original,
                'hypothesis_original': hypothesis_original,
                'reference_raw': reference_raw,
                'hypothesis_raw': hypothesis_raw,
                'reference_normalized': reference_normalized,
                'hypothesis_normalized': hypothesis_normalized,
                'raw_wer': sample_raw_wer,
                'raw_cer': sample_raw_cer,
                'normalized_wer': sample_norm_wer,
                'normalized_cer': sample_norm_cer
            }
            results.append(result)
            
            print(f"Reference (original):  {reference_original}")
            print(f"Hypothesis (original): {hypothesis_original}")
            print(f"Reference (normalized):  {reference_normalized}")
            print(f"Hypothesis (normalized): {hypothesis_normalized}")
            print(f"Raw WER: {sample_raw_wer:.2f}%, Raw CER: {sample_raw_cer:.2f}%")
            print(f"Normalized WER: {sample_norm_wer:.2f}%, Normalized CER: {sample_norm_cer:.2f}%")
        
        # Calculate overall raw metrics
        overall_raw_wer = wer(raw_references, raw_hypotheses) * 100
        overall_raw_cer = cer(raw_references, raw_hypotheses) * 100
        
        # Calculate overall normalized metrics
        overall_norm_wer = wer(normalized_references, normalized_hypotheses) * 100
        overall_norm_cer = cer(normalized_references, normalized_hypotheses) * 100
        
        # Save results to text file
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("BENGALI ASR EVALUATION RESULTS (SPRING-INX Model with Indic NLP)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("OVERALL METRICS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Raw WER:        {overall_raw_wer:.2f}%\n")
            f.write(f"Raw CER:        {overall_raw_cer:.2f}%\n")
            f.write(f"Normalized WER: {overall_norm_wer:.2f}%\n")
            f.write(f"Normalized CER: {overall_norm_cer:.2f}%\n")
            f.write(f"Total samples:  {len(dataset)}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            for idx, result in enumerate(results, 1):
                f.write(f"Sample {idx}:\n")
                f.write(f"Audio: {result['audio']}\n\n")
                
                f.write(f"Reference (original):  {result['reference_original']}\n")
                f.write(f"Hypothesis (original): {result['hypothesis_original']}\n\n")
                
                f.write(f"Reference (raw):  {result['reference_raw']}\n")
                f.write(f"Hypothesis (raw): {result['hypothesis_raw']}\n")
                f.write(f"Raw WER: {result['raw_wer']:.2f}%, Raw CER: {result['raw_cer']:.2f}%\n\n")
                
                f.write(f"Reference (normalized):  {result['reference_normalized']}\n")
                f.write(f"Hypothesis (normalized): {result['hypothesis_normalized']}\n")
                f.write(f"Normalized WER: {result['normalized_wer']:.2f}%, Normalized CER: {result['normalized_cer']:.2f}%\n")
                
                f.write("-" * 80 + "\n\n")
        
        # Save results to CSV file
        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'Audio File',
                'Reference (Original)',
                'Hypothesis (Original)',
                'Reference (Raw)',
                'Hypothesis (Raw)',
                'Raw WER (%)',
                'Raw CER (%)',
                'Reference (Normalized)',
                'Hypothesis (Normalized)',
                'Normalized WER (%)',
                'Normalized CER (%)'
            ])
            
            # Write data rows
            for result in results:
                writer.writerow([
                    result['audio'],
                    result['reference_original'],
                    result['hypothesis_original'],
                    result['reference_raw'],
                    result['hypothesis_raw'],
                    f"{result['raw_wer']:.2f}",
                    f"{result['raw_cer']:.2f}",
                    result['reference_normalized'],
                    result['hypothesis_normalized'],
                    f"{result['normalized_wer']:.2f}",
                    f"{result['normalized_cer']:.2f}"
                ])
            
            # Write summary rows
            writer.writerow([])
            writer.writerow(['OVERALL METRICS'])
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Overall Raw WER (%)', f"{overall_raw_wer:.2f}"])
            writer.writerow(['Overall Raw CER (%)', f"{overall_raw_cer:.2f}"])
            writer.writerow(['Overall Normalized WER (%)', f"{overall_norm_wer:.2f}"])
            writer.writerow(['Overall Normalized CER (%)', f"{overall_norm_cer:.2f}"])
            writer.writerow(['Total Samples', len(dataset)])
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
        print(f"\nRAW METRICS:")
        print(f"  Word Error Rate (WER):      {overall_raw_wer:.2f}%")
        print(f"  Character Error Rate (CER): {overall_raw_cer:.2f}%")
        print(f"\nNORMALIZED METRICS (Indic NLP):")
        print(f"  Word Error Rate (WER):      {overall_norm_wer:.2f}%")
        print(f"  Character Error Rate (CER): {overall_norm_cer:.2f}%")
        print(f"\nDetailed results saved to:")
        print(f"  - Text file: {output_txt}")
        print(f"  - CSV file:  {output_csv}")
        
        return {
            'raw_wer': overall_raw_wer,
            'raw_cer': overall_raw_cer,
            'normalized_wer': overall_norm_wer,
            'normalized_cer': overall_norm_cer,
            'results': results
        }


def main():
    """Main execution function."""
    
    # ===== CONFIGURE THESE PATHS =====
    MODEL_PATH = "D:\\SPIRE Internship\\SPRING_INX_wav2vec2_Bengali.pt"
    JSON_PATH = "D:\\SPIRE Internship\\dataset_all2.json"
    OUTPUT_TXT = "evaluation_results_indic.txt"
    OUTPUT_CSV = "evaluation_results_indic.csv"
    
    # Memory settings
    USE_CPU = True  # Set to True to use CPU instead of GPU (recommended for 4GB GPU)
    # =================================
    
    # Initialize evaluator
    print("Initializing Bengali ASR Evaluator with Indic NLP...")
    evaluator = BengaliASREvaluator(MODEL_PATH, use_cpu=USE_CPU)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(JSON_PATH, OUTPUT_TXT, OUTPUT_CSV)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("RAW METRICS:")
    print(f"  Word Error Rate (WER):      {results['raw_wer']:.2f}%")
    print(f"  Character Error Rate (CER): {results['raw_cer']:.2f}%")
    print("\nNORMALIZED METRICS (Indic NLP):")
    print(f"  Word Error Rate (WER):      {results['normalized_wer']:.2f}%")
    print(f"  Character Error Rate (CER): {results['normalized_cer']:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
