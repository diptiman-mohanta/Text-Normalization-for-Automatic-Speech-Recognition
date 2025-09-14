# Text-Normalization-for-Automatic-Speech-Recognition

---
## Paper 1: Thutmose Tagger: Single-pass neural model for Inverse Text Normalization

- Authors: Alexandra Antonova, Evelina Bakhturina, and Boris Ginsburg
- Published On: 
- The authors states that traditional methods like rule-based Weighted Finite-State Transducers (WFST) lack context awareness and are hard to maintain, while neural sequence-to-sequence (seq2seq) models are prone to hallucinations.
- The key innovation is treating ITN as a tagging task rather than a translation task.The goal of
tagging is to assign a tag to each input word in the spoken domain sentence so that the      concatenation of these tags yields the desired written domain sentence. This uses a single-pass token classifier based on a pretrained BERT model, making it simpler, more interpretable, and less error-prone than seq2seq models.
- This model is inspired from the LasserTagger. It shows that many monotonic sequence-to-sequence transformation task, such as text simplification or grammar correction are reformulated as tagging task.
- The model achieves state-of-the-art (SOTA) sentence accuracy on the Google Text Normalization (GTN) dataset for both English and Russian, with fewer unrecoverable errors.
- Model Architecture: A BERT-based encoder with a multi-layer perceptron (MLP) and softmax for token classification. It processes the entire spoken sentence in one pass, assigning tags to each input token.

### Proposed Approach:
- Initial data: To train this model they alighedn the GTN dataset. The GTN dataset consists of
unnormalized (i.e. written form) and normalized (i.e. spoken form) sentence pairs that are aligned on a phrase-level. To get  amonotonic one to one correspondence between each spoken word and corresponding fragments in writtten from the dataset is alligned into more granular level.
- Alignment: All corresponding phrases are extracted to create a parallel corpus for each semiotic class. Used Giza++ for this allignment. To do this alignment they tokenize the data first The spoken text istokenized by word boundary, while the written part is tokenized as follows: 1) All alphabetic sequences are separate tokens, 2) In numeric sequences each character is a separate token. 3) All non-alphanumeric characters are separate tokens. Additionally, we add an underscore symbol to mark the beginning and end of a sequence for future detokenization. For example, "jan 30,2005" is tokenized as "_jan_ _3 0 , 2 0 0 5_". Then they joined thogether the character token in the written part that are aligned to the same spoken input word. If a spoken input word aligns to nothing,  added a **DELETE** tag.
- Non-Monotonic alignment: One important restriction of the tagger mode is that spoken and written pairs are assumed to be monotonically aligned. Most of the spoken-written pairs in the dataset satisfy the requirement. Detected the non-monotonic examples and discarded them.
- Tag Vocabulary and Training Dataset: The alignment procedure splits the written sentences into fragme ts and align them one-to-one to be spoken form. Then count the frequencies of such written fragments and include some predefinec number of the most frequent of them in the tag vocabulary.  **SELF** and **DELETE** tags are also added to the tag vocabularies to signify that an input token should be either copied or deleted. The tagger model sees the whole sentence to make context dependent decisions. All input tokens outside ITN spans are mapped to **SELF** tags during data set creation.
- Post Processing: To get the final output applied a simple post processing procedure. specifically substituted  **SELF** tokens with input words,remove **DELETE** tokens, move tokens that have movements encoded in tags, and, finally, remove spaces between fragments bordered with under score symbols.

### Experiments:
- As baseline Duplex Text Normalization model is used to compare with the proposed model.
- Training Details: As backbone for this model prtrained model from HuggingFace library ***bert-base-uncased*** and ***distilbertbase-uncased*** for english and ***DeepPavlov/rubert-base-cased*** and ***distilbert-base-multilingual-cased*** for russian.
- Trained on 8V100 16GB GPU for 6 epochs using batch size 64, optimizer AdamW with 2e-4 learning rate, 0.1 warmup, and 0.1 weight decay.
- Used three metrics **1.Sentence accuracy**- An automatic metric that matches each prediction with multiple possible variable of the references. The errors are divided into two catagories **i.digit error: occurs when at least one digit differs from the closed reference variant** and **ii.other error: occurs when there is a non digit error is present in the prediction.** **2.Word Error Rate**- an automatic metric commonly used in ASR where each prediction is compared with ecactly one reference from the initial corpus. **3.Number of unrecoverable errors**- this shows the number of errors that corrupts the semantic of the input.

### Results:
- Duplex and Thutmose tagger models show similar results on "digit error" for English and Russian. At the same time, the Thutmoe tagger outperforms duplex by 1% and 3% sentence accuracy on defult and hard Russian test set.
- This model shows a slightly worse WER(+0.8%) on English defult test set, but othet metrics are better on the same test set.
- The difference between Thutmose tagger with BERT and DistilBERT for English is small: 0.07% and 0.46% sentence accuracy decreases for the defult and hard test sets respectively. For russian this difference is bigger: 0.73% and 2.28% respecively, since multilingual distilBERT is not specifically tuned for russian language.

### Error Analysis

Mostly from corpus alignment issues, e.g., digit duplications ("twelve thousand seventy one" → "120071" instead of "12071") and ("five million croatian kuans"→ "5 million million czk" instead of "5 million hrk" as on of the cause is hrk tag is too rare in the dictionary)

---
## Paper 2: A unified transformer-based framework for duplex test normalization

- Authors: Tuan Manh Lai, Yang Zhang, Evelina Bakhturina, Boris Ginsburg, and Heng Ji
- Published On:
- This "duplex" approach addresses the inefficiency of maintaining separate models for TN (in TTS systems) and ITN (in ASR systems), reducing complexity in production spoken dialog systems.
- Model Architecture: It contains two components—a tagger for span detection and a normalizer for conversion—built on pretrained transformers.

### Methodology

- Task Indicator: to apply duplex mode they append a task indicator (e.g., "TN" for written input, "ITN" for spoken).
- Transformer-Based Tagger: given orginal input sequence *T* = (t<sub>1</sub>,t<sub>2</sub>,......,t<sub>n</sub>), a task indicator token t<sub>0</sub> (t<sub>0</sub> ∈ {TN,ITN}) is added to the begining. Then the actual input sequence become (t<sub>0</sub>,t<sub>1</sub>,t<sub>2</sub>,......,t<sub>n</sub>). The role of the tagger is to predict a sequence of labels (y<sub>0</sub>,y<sub>1</sub>,y<sub>2</sub>,......,y<sub>n</sub>), where y<sub>i</sub> is th label corresponding to toke t<sub>i</sub>. The labels are {B,I}-TASK (indicator), {B,I}-SAME (keep as-is), {B,I}-PUNCT (punctuation), {B,I}-TRANSFORM(A semiotic span). This tagger first forms a contextualized representation for each input token using a transformer encoder like BERT. Then this is feed into a softmax layer to classify over the tagging labels. and to train the tagger a cross entropy loss function is used.
- Transfomer-Based Normalizer: Let *S* = (s<sub>1</sub>,s<sub>2</sub>,......,s<sub>m</sub>) be the set of all (predicted) semiotic spans in the input sequence T. *m* denotes the no of semiotic spans. The role of the normalizer is to transform each semiotic span into its appropriate from. For each semiotic span, input includes: task indicator + left context + "<m> span </m>" + right context (special tokens <m> </m> highlight the span).First, an input sequence of tokens is mapped into a sequence of input embeddings, which is then passed into the encoder. The encoder consists of a stack of Transformer layers that map the sequence of input embeddings into a sequence of feature vectors. The decoder is also Transformer-based. It produces
an output sequence in auto-regressive manner: at each output time-step, the decoder attends to the encoder’s output sequence and to its previous outputs to predict the next output token.This model i trained using a standard maximum likelihood i.e. using teacher forcing and a cross entropy loss. To make this tagger more reboust noisy spans are added to it. Combine normalized spans with unchanged parts (SAME/PUNCT).

### Experiments

- For english and russian standard google TN dataset is used for germany used a cleaned dataset from Spoken Wikipedia Corpora. For English a distilled version of RoBERTa for tagger and T5base for the normalizer. For German, a german version of BERT for tagger, and multilingual mT5 for normalizer.  For Russian, distilled version of the multilingual version of BERT for tagger, and distilled version of the multilingual version of mT5 for normalizer.
- Compared with baselines such as RNN-based Sliding window model, Transformer-based Seq2Seq Model, Nemo's WFSTs.

### Results

Achieves SOTA sentence-level accuracy on Google TN dataset (EN: TN 98.36%, ITN 93.17%; RU: TN 96.21%, ITN 85.67%). Also strong on a new German dataset from Spoken Wikipedia (TN 94.34%, ITN 87.71%) and an internal English TN dataset (>95% without fine-tuning).

### Error Analysis

They havemanually analyzed the errors made by English duplex system for TN. Among 7551 test instances, our model makes mistakes in 124 cases(1.64%). However, 113 of the cases are acceptable errors, and only 11 cases(0.146%)are unrecoverable errors. Among the 11 unrecoverable errors, seven are related to URLs, three are related to numbers,and one is miscellaneous.

---

## Paper 3: What is lost in Normalization? Exploring Pitfalls in Multilingual ASR Model Evaluations

- Authors: Kavya Manohar, Leena G Pillai, and Elizabeth Sherly
- This paper is critically examines the evaluation of multilingual Automatic Speech Recognition (ASR) models, with a strong emphasis on Indic language scripts (e.g., Hindi, Tamil, Malayalam).
- The core argument is that current text normalization practices—intended to standardize ASR outputs for fair performance metrics like Word Error Rate (WER)—are flawed for non-Latin scripts, leading to distorted text representations and artificially inflated performance scores on benchmarks.
- A proper text normalization routine is required to minimize penalization of non-semantic differences by aligning the predicted output more closely with the ground truth.

### Background and Related works
- Most ASR systems were trained on normalized text transcripts and produced output without punctuation and casing. Wisper gives output as UTF-8 text, requiring a comprehensive normalization process to accurately evaluate its performance. This ensure that the evaluation metric, WER penalizes only actual word mis-transcriptions, not formatting or punction differences.
- Wishpers normalization routine for english extends beyond basic casing and punctuation, incorporating transfromation such as converting contracted abbreviation to expanded forms and expanding currency symbols.
- this type of approach requires a language-specifiv set of transformation for non english text. Due to the lack of linguistic knowledge to develop such normalization it relies on a basic data driven approach and this inadvertently removes the vowel signs(matras). and these vowles signs are essential for correct word formation and pronounciation.

### Methodology

#### Analysis of Text Similarity after Whisper Normalization
- For emirircally assesment the impact of normalization different language they selected Latin script (English and Finnish), Indic scripts(Hindi, Tamil, and Malayalam), and South East Asian scripts (Thai).
- The METEOR score was employed to quantify the similarity between the original and normalized sentences.
- The score obtained are English=0.97, Finnish=0.95, Hindi=0.38, Tamil, Malayalam, and Thai = 0.00.

#### Impact of Whisper Normalization on WER
- To empirically analyze the impact of the normalization on the WER, the result of evaluation the orginal whisper small mode, reffread to as the baseline model, with and without the application of whisper's normalization on the test split of Google Multilingual speech dataset.
- The WER of the baseline model is significantly high for language other than English and Finnish with value of 86.95% for hindi, 93.3% for tamil and 287.4% for Malayalam.
- While the application of the Whisper's normalization results in modest WER improvements for English and Finnish, with an absolute reduction of 5.1% and 3.2% respectively, Indic language experience suspicious absolute WER reductions: 21.9% for Hindi, 41.5% for Tamil and a substantial 152.2% for Malayalam.
- Due to this poor perfomance they conducted a further comparision of WER with and without Whisper's normalization on publicly availabe model that have been derieved from the baseline model after language-specific fine tuning. This improved the performance of the Hindi, Tamil and malayalam model with the absolute reduction in WER with the decrease of 10.7% for Hindi, 21.3% for Tamil and 34.1% of Malayalam.

