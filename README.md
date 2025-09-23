# Text-Normalization-for-Automatic-Speech-Recognition

# Date: 12 Sept 2025 
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

# Date: 13 Sept 2025 
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

# Date: 15 Sept 2025 
---
## Mark My Words: A Robust Multilingual Model for punctuation in Test and Speech Transcripts
- Authors: Sidharth Pulipaka, Sparsh Jain, Ashwin Shankar, Raj Dabre
- Punctuation plays a vital role in structuring meaning to a text or speech.
- In this model, they introduced cadence, a generalist punctuationt restoration model adapted from pretrained LLMs.
- Cadence is designed to handel both clean written test and highly spontaneous spoken trasnscripts. It surpasses the previous SOTA in performance while expanding support from 14 to all 22 Indian Language and English.
- Indic languages face a substantail hurdles. They include scarcity of annoted corpora, especially for low-resources languages and linguistic complexity with diverse scripts, grammars and unique marks. To bridge this gap for Indic Languages, they constucted a diverse multilingual punctuation corpus covering both written and ASR-transcribed text.
- This corpus is aggregated from multiple sources, including Sangraha-verified, IndicVoices, TranslatedCosmopedia and IndicCorp-v2. They adopted Geema3-1B-pretrained model for punctuation restoration by converting it into a bidirectional transformer using a masked next token prediction.
- This model support English and all 22 scheduled languages of India.

### Methodology

#### Data Strategy for multilingual punctuation restoration.
- For intial continual pre training phase, they leverage large-scale high-quality multilingual web corpora. These resources are selected for their broad linguistic coverage, providing the model with exposure to a wide arrey of languages and writting styles.
- To prepare the model specifically for punctuation restoration, they construct a substantial and hetrogenious finetuning dataset through a significant data aggregation efforts.

#### Model training and Adoptation
- Started with pre-trained transformerbased language model these are deighend for unidirectional text generation processing context only from preceding token. For sequence tagging task like punctuation restoration bidirectional infromation flow is highly benificial and adopted a model's attention mechanism to be fully bidirectional.
- Instead of conventioal masked language modeling, they used a Masked Next Token Prediction. In this modified setup, a random subset of tokens in an input sequence is masked. The model's predictive task is then specifically focused: for an unmaksed token at the position *i* if its subsequent token position is *i*+1 is the maksed, the mode is trained to predict this masked token *i*+1.  This prediciton is prefromed using the contextual representation of the token at position *i*.
- Initially trained on a high resources language to establish a robust foundational representations. Then this model is exposed to a mid resouces language which allowed the model to begin generalizing across the related linguistic structures and benifit from these larger datasets. Then low resources languages are introduced which encouraged knowlege transfer from the more data rich languages learned in the previous phase and achieved good performance on low data languages. Then model is trained on a mixture of datas from all supported languages.
- Then the model is finetuned specifically for the punctuation restoration task using amalgamated dataset described earlier.

### Results
- Achieved an overall socre of 0.7924 on written text and 0.6249 on spontaneous speech transcripts. On complete set of 30 supported punctuation labels it achived overall score of 0.5931 on written text and 0.4508 on spontaneous speech transcripts.

---

## Paper 5:indic-punct: An automatic punctuation restoration and inverse text normalization framework for Indic languages
- Author: Anirudh Gupta, Neeraj Chhimwal, Ankur Dhuriya, Rishabh Gaur, Priyanshi Shah, Harveen Singh Chadha, Vivek Raghavan
- ASR generates text which is most of the time devoid of any punctuations.
- An approach for automatic punctuation of textt using a pretrained IndicBERT model. Inverse text normalization is done by hand writting weighted finite state transducer grammer.
- Developed this tool for 11 Indian languages.
- Prior approaches to automatic punctuation have used lexical anf prodsody features or a combination of both.

### Framework Parts
- Contains two parts: Punctuation restoration and Inverse Text Normalization

#### Inverse Text Normalization
- classify: It creates a linear automation from the input string and composes it with the final classification WFST, which transduces numbers and inserts semantic tags.
- Parse: Parsed the tagged string into a list of key value items representing  the different semiotic tokens.
- Generate recording: It is a generator function which takes the parsed token and generated string serialization with different reordering of the key value items.
- Verbalize: It takes the intermediate string representatuon and compose it eiht the final verbalization WFST.

#### Punctuation Restoration
- Dataset Preperation: Used IndicCorp dataset and created a own training data by filtering out line wich containes some punctuations(considered 3 punctuation for all the languages sentence end, comma and question mark). Lines are normalised using IndicNLP. Removal of words of other languages.Once clean text are obtained line of the test are prepared for data loading.
- Model Architecture and Training: They posed as this an token classification tast and used IndicBERT for that downstreaming task.

### Results 
Calculated macro F1 score across labels for all languages and the score vary from 0.77-0.86 for all languages.

---
# Date: 23 Sept 2025

## Paper 6: Four-in-One: A joint approach to inverse test normalization, punctuation, capitalization and disfluency for Automatic Speech Recognition
- Authors: Sharman Tan, Piyush Behre, Nick Kibre, Issac Alphonso, Shuangyu Chang
- Converting ASR output into written from involves applying features such as inverse text normalization (ITN), Punctuation, capitalization and disfluency removal.
- ITN formay entities such as number,dates, times, and addresses.
- Introduced a novel stage approach to spokekn to written text conversion consisting of a single joint tagging model followed by a tag application stage.
- Defined a text processing pipeline for spoken and written form public datasets to jointly predict token level ITN, Punctuation, capialization and disfluency tags, as described in section 3 and 4.

### Proposed Method

#### Joint labeling of ITN, Punctuation, Capitalizatiom amd disfluency
- This adresses ASR fomatting as a multiple sequence labeling problem. First tokenize the spoken form text and then use a transformer encoder to learn a shared representation of the input.
- Four task-specific classification heads-corresponding to ITN, Puncutation, capitalization and disfluency- predict four token level tag sequence from the shared representation.
- Each classification head consists of a dropout layer followed by a fully connected layer. And Used the cross entropy loss function and jointly optimizes all four tasks by minimizing an evenly weighted combination of the lossws as shown in: $CE_{\text{joint}} = (CE_i + CE_p + CE_c + CE_d)/4$, where CE_i, CE_p, CE_c, CE_d are the cross entropy of function fot ITN, Punctuation, capitalization and disfluency respectively.

#### Tag application
- The Four tag sequences to format the spoken form ASR output as their writtenf form. As the tag sequences are token level, they are converted them into word-level for tag application.
- To format the ITN entities, extracted each span of ITN token that are consecutively tagged as the same ITN entity type and span. Then applied WFST grammar for that entity type to generate the written form.
- ITN formatting may change the no of words in the sequence so they preserved the alignments between the orginal spoken form tokens and formatted ITN entities. When multiple spoken-form token maps to a single WFST output, only applied the last punctuation tag and the first capitalization tag.
- For puncutation appended the indicated punctions tags to the corresponding words.
- For capitalization, they capitalize the first letter or entirety of word, as tagged.
- To remove disfluencied they removed the disfluency tagged words from the text sequence.
- compared both task specific and joint models for using four independent task specific models in the real scenarios may result in undesirable conflicts between features.

### Data Processing Pipeline

#### Dataset

Use publicly availabe dataset from various domains as well as additional set specifically targeting ITN and disfluency:
| Dataset | Distribution |
| :--- | :--- |
| OpenWebText | 22.8% |
| Stack Exchange | 13.6% |
| OpenSubtitles2016 | 3.3% |
| MAEC | 2.9% |
| NPR Podcast | 0.6% |
| SwDA + SWB + Fisher | 0.3% |
| Web-crawled ITN | 56.4% |
| Conversational Disfluency | 0.1% |

#### Data processing written form
- Apart fro  SwDA, SWB, and fisher all corpora are written form text containing ITN, Punctuation and capitalization.
- To jointly train a single model to predict the tag sequences corresponding to ITN, punctuation, captilization and disfluency, processed each of the set to contain token-level tags for each of the four tasks.
- Filtern and cleaning the dataset by preserving natual sentence or paragraph as rows and removing characters apart from the alphanumeric, puncuation and necessary mid-word symbols such as hyphens.
- To generate the spoken form equivalent of the written from dataset, they used WSFT grammar based text normalization.

#### Data processing of spoken form
- SwDA, SWB, and Fisher are spoken-form coversational transcription and thus dont contain ITN, capitalization or punctuation.
- This data is converted to written format form by applying a commercial formatting service. Then generate ITN, capitalization and punctuation tags using the same process as written form datasets.
- SwDA containes dialog act annotations so it is translated to token level disfluency tags.

#### Tag Classes
- ITN: Tag each token as one of the five entity types (alphanumeric, numeric, ordinal, money, time) or "0" representing non-ITN. As WFST grammers is applied on the each spoken form ITN entity span, signified each ITN entity span by tagging the first token as the entity tag and prepending an undersocre character to the remaining tags in the span.
- Punctuation: Defined 4 tag catagories: comma, period, question mark, and "0" for no punctuation.
- Capitalization: defined 3 tag categories: all uppercase("U"), capitalize only the first letter ("C") and all lowercase ("O").
- Disfluency: Followinf SwDA annotation they are defined into 7 catagories: correction of repetation, reparandum repetation, correction, reparandun, filler word, all other disfluency and non-disfluency.

### Result
### Punctuation results:

| Test Set | Model | COMMA `P R F1` | PERIOD `P R F1` | Q-MARK `P R F1` | OVERALL `P R F1` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Ref. CNN Stories | TASK-SMALL | 84 80 82 | 90 83 86 | 86 83 84 | 86 81 84 |
| | TASK | **84** **82** **83** | 91 84 87 | 86 **85** **85** | **87** **83** **85** |
| | JOINT | 84 81 82 | **90** **84** **87** | **86** 83 **85** | 86 82 84 |
| Ref. DailyMail Stories | TASK-SMALL | 76 79 77 | 90 88 89 | 82 71 76 | 82 83 82 |
| | TASK | 77 **80** **79** | **92** **90** **91** | **88** **78** **83** | 84 85 84 |
| | JOINT | 77 79 78 | 91 89 90 | 84 74 78 | **83** **84** **83** |
| ASR IWSLT 2011 TED | TASK-SMALL | 66 31 43 | 73 68 71 | 59 42 49 | 69 49 56 |
| | TASK | 68 **33** **44** | 75 68 72 | 56 45 **50** | 71 50 **57** |
| | JOINT | 70 **20** 31 | 75 67 71 | **80** 26 39 | **73** 42 50 |
| Ref. IWSLT 2011 TED | TASK-SMALL | 78 61 69 | 81 88 85 | 80 85 82 | 79 74 77 |
| | TASK | 79 **67** **72** | 84 **88** **86** | 71 **90** **80** | **81** **77** **79** |
| | JOINT | 79 63 70 | 82 87 85 | **80** **85** **82** | 80 75 77 |
| ASR NPR Podcasts | TASK-SMALL | 71 60 65 | 83 77 80 | 80 68 74 | 77 69 73 |
| | TASK | 71 **62** **67** | 84 **78** **81** | 80 68 74 | 78 70 74 |
| | JOINT | **71** 60 65 | 83 77 80 | **82** **69** **75** | **78** **69** **73** |
| ASR Dictation | TASK-SMALL | 69 54 61 | 72 78 75 | 48 94 **64** | 70 65 67 |
| | TASK | 70 **57** **63** | 73 79 76 | 44 94 60 | 71 67 **69** |
| | JOINT | **70** 56 62 | **73** **80** **76** | **50** 81 62 | **71** **67** 68 |
| Ref. Dictation | TASK-SMALL | 73 59 65 | 82 76 79 | 71 92 80 | 77 66 71 |
| | TASK | 73 61 66 | 83 **78** 80 | 65 100 79 | 77 68 72 |
| | JOINT | **73** **61** **66** | **85** **77** **81** | **72** **100** **84** | **78** **68** **72** |

### ITN results:

| Test Set | Model | ITN `P R F₁` |
| :--- | :--- | :---: |
| Ref. CNN Stories | TASK-SMALL | 88 87 88 |
| | TASK | 88 87 88 |
| | **JOINT** | **89 87 88** |
| Ref. DailyMail Stories | TASK-SMALL | 84 84 84 |
| | TASK | 84 84 84 |
| | **JOINT** | **85 84 85** |
| ASR NPR Podcasts | TASK-SMALL | 76 58 66 |
| | TASK | 77 59 66 |
| | **JOINT** | **77 59 67** |
| Ref. Wikipedia | TASK-SMALL | **65** 69 **67** |
| | TASK | 63 68 66 |
| | **JOINT** | 64 **69** 66 |
| ASR Dictation | TASK-SMALL | 75 59 66 |
| | TASK | 74 58 65 |
| | **JOINT** | **76 60 67** |
| Ref. Dictation | TASK-SMALL | 84 62 72 |
| | TASK | 83 62 71 |
| | **JOINT** | **84 63 72** |
| Ref. Web-crawled ITN | TASK-SMALL | 82 76 79 |
| | TASK | 85 75 78 |
| | **JOINT** | **82** 76 **79** |

### Disfluency results:

| Test Set | Model | DISFLUENCY `P R F₁` |
| :--- | :--- | :---: |
| Ref. SwDA | TASK-DISF | **95 84 89** |
| | TASK | 89 47 62 |
| | **JOINT** | 94 **85** **89** |
| Ref. Conv. Disfluency | TASK-DISF | **78 44 56** |
| | TASK | 72 20 32 |
| | **JOINT** | 76 42 54 |

### Capitalization results
*Uppercase refers to words longer than 1 letter that are uppercase, Capital refers to words with only first letter capitalized, and Single-case refers to 1-letter words that are uppercase.*

| Test Set | Model | UPPERCASE `P R F₁` | CAPITAL `P R F₁` | SINGLE-CASE `P R F₁` | OVERALL `P R F₁` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Ref. CNN Stories | TASK-SMALL | 38 83 52 | 94 92 93 | 97 49 66 | 93 84 87 |
| | TASK | 38 83 52 | 94 93 94 | 97 49 66 | 93 85 88 |
| | **JOINT** | **38 83 53** | **95 93 94** | **97 49 66** | **93 85 88** |
| Ref. DailyMail Stories | TASK-SMALL | 82 92 87 | 94 93 93 | 91 77 84 | 93 92 92 |
| | TASK | 81 **93** 87 | 94 94 94 | 92 77 84 | 93 93 93 |
| | **JOINT** | 82 92 87 | **95 94 94** | **92 77 84** | **94 93 93** |
| ASR NPR Podcasts | TASK-SMALL | 80 68 74 | 88 83 86 | 93 **84 88** | 88 82 86 |
| | TASK | 83 69 75 | 89 84 86 | 90 81 85 | 89 83 85 |
| | **JOINT** | **83 69 75** | **89 84 86** | 93 82 87 | **89 83 86** |
| ASR Dictation | TASK-SMALL | 75 74 74 | 79 82 81 | 60 72 65 | 78 81 80 |
| | TASK | 73 77 75 | 80 83 81 | 54 64 59 | 78 82 79 |
| | **JOINT** | **75 79 77** | **80 83 82** | **63 77 69** | **79 82 81** |
| Ref. Dictation | TASK-SMALL | 77 88 82 | 88 83 85 | 72 53 61 | 86 82 84 |
| | TASK | 75 88 81 | 87 **85** 86 | 72 53 61 | 85 84 84 |
| | **JOINT** | **78 88 83** | **88 84 86** | **73 56 63** | **86 83 85** |

---

## Paper 7: End-to-End joint puncutuated and normalized ASR with a limited amount of punctuated training data
- Authors: Can Cui, Imran Sheikh, Mostafa Sadeghi, Emmanuel Vincent
- Dataset used: LibriSpeech 
- Transcribing speech into text with punctuation and casing has been active area of research in automatic speech recognition.
- Joint punctuated and normalised ASR, which produces transcripts both with and without punctuation ans casing is highly desirable because it improves human readablity and it extends compatiblity with the NLP models that either exploits or discard punctuation information and if it simplify model deployment and maintance.
- Conventional punctuated ASR approach post-processes normalized ASR output using a punctuation and case restoration model sich as modified recurrent neural network transducer ASR with a finetuned language model.
- Some other alternative is end to end ASR, which is directly generated punctuated transcripts. these modes are less accurate in word recognition that equal-sized normalized ASR models, leading to poorer normalized ASR perfomance and increased computation time for output normalization.
- This paper proposes an E2E joint punctuated and normalized ASR system that is effiecient in both task , trainable with limited punctuated labeled data and suitable for streaming. Intoduced and compared two complementary approached to train a stateless transducer-based E2E joint punctuated and normalized ASR model.,
- First approach uses an LM to generate punctuated trainig transcripts. As such LMs may not be accurate enough or availabe for certain domains. So a second approach is proposed in which a single decoder is conditioned on the type of output.

### Preliminary of stateless transducer-based punctuated ASR
- An E2E punctuated ASR system transcribe speech into a punctuated transcripts.
- Given an acoustic feature sequence  $$X \in \mathbb{R}^{L \times A}$$ , where L is the sequence length and A the feature dimension , the training objective is to maximize the probablity.
- A stateless transducer-based E2E ASR system, which uses RNN-T framework with a stateless prediction network can be readily extended to the punctuated transcription task, a stateless transformer encoder with downsampling and a zip like structure further enhance efficiency while maintaining accuracy.

### Proposed Methods

#### 2-Decoder joint normalized ASR
- An ASR system with two decoders, each consisting of its own predictor and joiner. using the output of the same encoder these two decoder generate the punctuated and normalized transcripts.

#### Training ASR using the auto-punctuated transcripts
- Automatically generated punctuated transcripts of ASR training data can be used to train punctuated ASR model in the absence of human generated punctuated transcripts.
- This work focus on punctuation and casing. This punctuation and case enhanced transcripts can directly obtained from normalized transcripts using the available SOTA punctuation and case restoration models.

#### Conditioned predictor ASR
- The drawbacks of the autopunctuated transcription driven ASR training approach presented above is that errors made by the punctuation and case prediction model may propagate into the ASR models.
- In addition to this there are such cases and punctuation restoration model may not be accurate enough or even available for certain domains and or languages.
- The proposed approach efficetively utilizes the small amount of punctuated training data and the large amount of normalized training data by using a single conditioned predictor to handle both punctuated and normalized transcription mode ID, which is an input that specifies wheather we want a normalized(N) or punctuated(P) output.
-  The token embeddings are concatenated with the modes embedding before feeding them to the following predictor layers.
-  The conditioned predictor ASR system uses same loss fucntion as decoder only part of the input samples will have a punctuated reference transcript ans will use the corresponding loss. This lead to difference between the both mofrl perfomance and the output modes and particularly poos performance of the pucntuated transcription mode.

### Results
- Achives a better perfomance on out of domain test data with upto 17% relative punctuation-case-aware word error rate reduction. 
- The second approach use a single decoder conditioned on the type of output. This yeild a 42% relative PC-WER reduction comapred to whisper-base and 4% relative(normalized) wer reduction compared to the normalized output of a punctuated-only model.
- This proposed model demonstates the feasiblity of a joint ASR system using a little as demonstates the feasiblity of a joint ASR system as little as 5% punctuated training data witha moderate(2.4% absolute) PC-WER increase.

## Paper 8: Universal-2-TF: Robust All-Neural Text formatting for ASR


---
# Trying existing normalisation methods on both train and test transcripts and analyse ASR performance with and without normalisation


# Date: 18 Sept 2025 
---
## Steps
- Data Preparation: Alight the audio file and transcript pairs.
- Dataset splitting
- Copy a pair for raw transcripts (with out normalisation)
- Apply normalization to create the normalised version.
- Use a pretrained model.
- Evaluate using for (WER,CER,SER).
---
## Dataset preparation 
- We already have an main directory and inside it have many different folder each folder have some audio files named like this `IISc_RESPIN_hi_D1_40020_251001_F_AGRI_400208_40207448` and an text file which contains its ground truth like this : `IISc_RESPIN_hi_D1_40020_251001_F_BANK_401129_40200343 हाइब्रिड के मामलों में कफी मात्रा में ब्याज की वापसी भी की जाती है` `IISc_RESPIN_hi_D1_40020_251001_F_AGRI_400208_40207448 क्रेप चमेली अपनी खुसबू के लिए प्रसिद्ध है` `IISc_RESPIN_hi_D1_40020_251001_F_AGRI_403379_40207717 एक एकड़ में कितना पोटाश डालना चाहिए ?`.
- created a structured JSON `dataset_all.json` with `(audio_path, transcript)` pairs using this `datasetprep.ipynb` code.
---
## Dataset splitting
- Splitting the training dataset for 80-10-10 for train `train.json`, validation`valid.json`, and test files`test.json`.
- As i am using pretrained ASR models directly used `dataset_all.json` to test the models.
---
# Date: 22 sept 2025

Evaluated ASR perfomance on the Test dataset `dataset_all.json` using different pretrained ASR model like `indicwav2vec-hindi`,`Wav2Vec2-large-xlsr-hindi`,`vakyansh-wav2vec2-hindi-him-4200`, and `wav2vec2-large-xlsr-53` using the code `nonormalization.py` and all the results are stored in `Without Normalization Results` directory.
