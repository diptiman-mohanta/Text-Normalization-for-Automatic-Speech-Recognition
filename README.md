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
---
# Date: 24 sept 2025
## Paper 8: Universal-2-TF: Robust All-Neural Text formatting for ASR
- Authors: Yash Khare, Taufiquzzaman Peyash, Andrea Vanzo, Takuya Yoshioka
- ASR systems produces text output in spoken form, requiring text formatting (TF) post processing to convert the ASR model's output into a written style. This enhance the readablity of the generated transcrits and improves compatiblity with various downstream processes.
- ASR model particularly seq2seq models like whispers are trained on audio files with written form transcripts collected from the Internet and can directly generate properly formatted transcipts, separating ASR to Speech-to-text and TF offers practical advantages making TF remain essential in modern commercial ASR systems.
- This papet presents a fully fledges all-neural TF method that performs PR, truecasing and ITN.
- The method compraises of two neural network models that work together to perfrom the tasks.
- The first model is a multi-objective token classifier that handels PR and capitalization while identifying textual spans that may require ITN or mixed-casing. The multi-bjective token classifier has a shared encoder or multiple heads which is demonstrated to reduce inference cost without accuracy loss.
- The second model is a seq2seq model applied to the identified spans to perform ITN and mixed-case word conversion. By limiting the text segment length processed by the seq2seq model the proposed method achieves a practically affordable computional cost and avoids catastrophic hallucination while benefitting from the enhanced flexiblity provided by the seq2seq model compared the WFST.

### Proposed Model
- The name is based on a pipeline comprising of two models, a multi-objective token classifier and a seq2seq model. Both models are based solely on text generated by STT, with no acoustics utilized.
- The first stage perfroms multi-objecitve token classification using a multihead model, predicting punctuation marks and token level casing labels while identifying textual spans requiring mixed casing and ITN.
- The second stage uses a seq2seq encoder-decoder model to convert short, unformated textual spans identified in the previous stage into their formatted expressions to jointly perform mixed-casing and ITN.
- This avoid full transcript processing provides two benefits. Firstly it focusing on restricted textual spans minimizes computational overhead and reduces processing time. Second, this approach allow the model to be fine tuned for specific ITN and mixed case conversion tasks, enhancing its effectiveness without handling sacrificing robustness in handling generic text conversion cases.

 #### Multi-Objective Token Classification
 - The first model is a multi objective network undertaking multiple token classification tasks. Given in input text, a sequence of feature representation is first obtained through a transformer-based encoder.
 - These representation are then fed into three linear heads to predict token-level labels, with each head focusing on different tasks through disjoint label sets, namely punctuation restoration, truecasing and ITN span identification.
 - An input token sequence provides by an STT model, each classification head predicts a label sequence of the same length.
- Then they trained the three headed model including the both encoder and the task specific classification head, by minimizing a combined loss function that aggregates cross entropy losses from each classification head.
- At the inference time the predicted labels from each heas are used for punctuation restoration, truecasing and ITN span identification.

#### Seq2seq Text Span conversion
- This languager model is based on an architecture using a bidirectional encoder and an autoregressive decoder.
- It is trained to perfrom ITN and convert lowercase words into properly mixed cased expressions.
- A sequence of tokens in each span identified from the first stage, the model predicts an output token sequence that maximizes a prosterior probablity, where the probablity distribution is learned during training.
- The biderectional encoder of the seeq2seq model computes hidden states which are used to conditioned the decoding process. This decoder generated the output sequence Y autoregressively using the greedy search startegy, that is the output toekns are produced one at a time.
- At each time step t, the decoder finds the token that maximizes the posterior probablity over the token vocabulary.
- The model is trained by using a dataset consisting of input output textual pairs.

### Result
The Universal-2-TF model gives PER:29%, CER:0.9%, M-WER:0.4%, I-WER: 30.3%. The Universal-1-TF model gives PER:29.9%, CER:1.2%, M-WER:0.6%, I-WER:52.7%. And the Full seq2seq gives PER:35%, CER:2.5%, M-WER:2.3%, I-WER:37.6 on the evaluation dataset. 

## Paper 9: Neural Inverse Text Normalization
- Authors: Monica Sunkara, Chaitanya Shivade, Sravan Bodapati, Katrin Kirchoff
- Dataset extracted from ParaCrawl, News commentary and MuST-C from German,Spanish and Indian Languages.
- The Finite Scale Transducer(FST) based approaches to ITN work well but they are expensive to scale across languages since native languages speakers are needed to curate transfromation rules.
- The data for ITN model trainig is not publicly availabe and hard to collect, we employ a data generation pipeline for ITN using a text to speech frontend.
- Proposed a neural ITN solution based on seq2seq models that can perform well across domains and noisy setting such as conversational data.

### Text Processing Pipeline
- During this process of data preparation found some of the major problems:
    - There are several variation of a written form of text, but text normalization techniques always use one fixed variation. this further increases with the varuation in locale and languages.
    - Text normalization systems often ignore several punctuation symbols since the goal is to produce spoken form of text. However, punctuation is often relevant in the written form.
    - Text normalization techniques may introduced errors when normalizing numbers or expanding some short forms, as the conversion requires to disambiguate based on the context.
- To adress these issues they create a synthetic data by randomly sampling from cardinal numbers related sentences and introducing additional variations required for modelling ITN in spoken form of text.

### Models
#### Finite State Transducer
- Conventional baseline approach is a finite state transducer constructed using JFLAP. Each state in the FST performs a series of edits to the input string to get its corresponding written format output string.
- This FST model covers a wide range of entities which do not require contextual understanding or disambiguation such as: Cardinals, Fractions, date, time etc
#### Neural ITN
- they modeled ITN as a seq2seq problem where the source and target are spoken and written form of text respectively.
- This seq2srq model uses Bahdanau content based additive attention to align the output sequence with input.
- Also implemented as non recurrent transformer based sequence to sequence model where multihead self attention layers are used in encoder and decoder. For all our transformer models, the source and target sentences are segmented into sub word sequence.
- **Copy Attention:** For ITN, there existing a significant overlap between source and target sentence. Standard seq2seq models which is rely on content bases attention are often prone to undesirable errors such as insertion, substitutions and deletion during conversion. The copy mechanism use a generation probablity to choose between source sentence vocabulary and a fixed target vocabulary thus allowing to generate out of vocabulary words.
- **Pretrained Models for ITN:** Recent pretrined model achived tremendous sucess and lead to significant on various natural languages understanding tasks. These models trained on aa large amounts of unlabelled data capture rich contextual representation of the input sentence. In this they attempted two strategies to incorporate pretraining into ITN:
  - To use a pretrained seq2seq model and finetune it for ITN: for this intialized the encoder decoder of a seq2seq model with a pretrained BART model and then finetuned the model on the ITN dataset.
  - Use a pretrained masked language model like BERT as context-aware embedding for ITN: Used BERT to extract context aware embeddings or fuse it into each layer for transfomer encoder or decoder via an attention mechanism.
- **Hybrid Solution:** seq2seq model is prone to errors and neural network need a good amount of data for training to handle new entity. Thus they propose a novel hybrid approach to combine neural ITNS with an FST, where the spoken form output of ASR system is first passes through the proposed neural ITN model followed by FST.A confidence score emitted by the Neural ITN model is used as a switch to make a decision in the run time weather to use neural ITN input.

### Results
- Neural ITN (Transformer + BERT-fusion + Copy attention) consistently outperforms rule-based FSTs.
- Copy attention is better for robustness to OOVs and out-of-domain data.
- BERT-fusion is better for in-domain normalization accuracy.
- A hybrid Neural + FST framework provides additional reliability for production systems.The approach generalizes well to multilingual settings without expert-crafted rules.
---

# Date: 25 Sept 2025
## Paper 10: Streaming, Fast and Accurate On-Device Inverse Text Normalization For Automatic Speech Recognition
- Authors: Yashesh Gaur, Nick Kibre, Jian Xue, Kangyuan Shu, Yuhui Wang, Issac Alphanso, Jinyu Li, Yifan Gong
- In this approach it consist of a new transformer-based tagger, which tags incoming lexical tokens from ASR in a streaming manner.
- The Tag contains information about the ITN category that can be associated with any tagged span.
- Once tagged span is availabe an ITN categories specific WFST performs the actual conversion, only on the tagged part of a sentence.
- The contributions of this papers are:
   - Propose a novel modelling solution for ITN. It splits the task into tagging and transduction. This alloweded to get high quality, streaming and light weight models that can be deployded to on device applications.
   - Designed a chunk based transformation tagger which enables streaming ITN. This design configured to trade off between accuracy and latency.
   - Proposed a tag to denote a particular ITN category.
 
### On-device Inverse Text Normalization

#### Modeling ITN within E2E ASR
- Conventionally, ITN has been modeled as separate components that resides in the post processing pipeline of the ASR.
- Then the end-to-end(E2E) training paradigm allows us to train a model that goes from speech to display fromat text directly. This means that ITN is learnt implicitly within the ASR model. In this ITN streaming comes out ina truly streaming fashion with no additional latency. The memory footprint of the model is reduced due to no external ITN model to store.
- Some drawback are since speech recognition is tightly coupled with ITN in this scenario, it looses all the flexibility in ITN system configuration. In the absence of external and configurable ITN, one would need to train a different E2E-ASR model for every domain. And updating any ITN need a lot of retraining if the ASR model on all data again.

#### Weighted Finite State Transducers with rescoring
- WFST are very nicely suited to the task of ITN beacuse ot allow application of the arbitary hand crafted rules and in many scenarios, can perform task in a compact or robust manner.
- Since spoken forms are ambiguous, we use the spoken-to-written FSTs to map them to multiple tagged written-form candidates and use a ranker to choose the optimal one depending on the context.
- The ranker is utilized in a simple log-linear interpolation setup.
- They choosed an n-gram model as a main choice for ranker. This is also built into the FST.
- An additional LSTM ranker to further improve contextual re-ranking.

#### Modeling ITN as a Seq2Seq task
- In this they considered the transformer-seq2seq architecture, to model the task of ITN.
- Learning the ITN task is an end-to-end manner has some advantages:
   - Unlike WSFTs, where experts needs to prepare the conversion rules, these models lear all the rules entirely from the data, with no involvement from human expert and can be scalable to new domains and languages with more and more data.
   - These models tens to be all-neural, their size can be compressed using a myriad of techniques.
- There are some challenges that prohibits the deployment of an all neural model for ITN.
  - ITN models are required to change their behaviour by ingesting arbitary human specific rules. This kind of functionality is difficult to enable with all neural model since they require a large amount of data to learn.
  - Even when trained with large amount of data, all neural ITN model can still suffer poor generalization.
 
#### Proposed approach: Transformer Tagger + WFST
- This aapproach is built on the insight that ITN can be broken into 2 disjoint step.
   - First step finds which parts of the sentence needs to be converted and corresponding ITN category for them.
   - Second step is actual conversion according to rule of ITN category.
- More specifically trained a transfomer neural network "tagger" which process the output of ASR in a streaming manner and predicts a "tag" for every input token. Each tag is associated with a certain ITN category. For token which dont belong to any ITN category, a blank token is put out.
- Once the tag are predicted, the WFST is responsible for the actual conversion. The WFST component is a collection of several FSTs, where each FST is responsible for a particular ITN category or tag.
- The transfomer-tagger learns to use the context, both history and limited feature, to predict what tags needs to be assigned to any input token.
- To make this transformer work in a streaming manner, they used a chunk based processing. More specifically, the transformer only processes a certaun chunk of token at a time nd it does not have acess to all the token in the future to make the prediction.
- Chunk based processing means that output is not available untill all tokens in a chunk are available.
- For streaming and on-device application, a very small latency is preferred. However a small chunk size means limited look ahead and a smaller window to do context modeling. Hence there is a trade-off between latency and accuracy.

### Result
- For text only evaluation this model outperform other baseline with the precision(0.81), recall(0.84) and recall(0.82) with the model size 5.5MB lowest of all the baselines(WFST+n-gram,WFST+n-gram+LSTM,S2S-small,S2S-large, Tagger+WFST).
- In speech to text evaluation this proposed model+ Lexical ASR outperfrom other baselines with Precision 0.71, Recall 0.75, F1 0.73 and TER 22.70.


## Paper 11: Mixed Case Contectual ASR using Capitalization Masks
- Authors: Diamantiono Caseiro, Pat Rondon, Quoc-Nam Le The, Petar Aleksic
- E2E mixed case ASR systems that directly predict words in the written domain are attractive due to being simple to build, not requiring explicit capitalization models, allowing streaming capitalization without additional effort beyond that required for streaming ASR and their small size.
- Mixed case E2E ASR model are trained using written domain training utterances with case information and different case variants of the same word are treated as independent token to be predicted by ASR.
- Two major advantages are:
    - They are easy to build.
    - Explicit capitalization module is not required during inference.
- One major disadvantage is its higher word error rate(WER) and capitalization quality may be inferior to using an explicit capitalization model.
- In this paper they proposed a novel representation for mixed case words that enables the use of single case biasing model and achieves similar ASR quality as mixed case ASR, while achieving improved capitalization quality and reduced size and compilation time of contextual models.

### Capitalization Masks
- Proposed the use of capitalization masks to overcome the contextual modeling problems. These masks allow us to decouple the word from its capitalization information, enabling us to compactly model allcase variants and avoid redundancy because all capitalized variants of an word decompose into the same peices.
- With this mask a word is represented in two parts:
   - The first part is a single case version of the word.
   - The second part is optional and consists of a sequence of bit 0/1 indicating if the character at the corresponding position in the word is upper(1) or lowercase(0).
- Used 16 Unicode Braille charcters as our control characters, representing nibbles of values 0 through 15 in order. These allow for easy interpretation of mask because the presence of one or two dots in each row represents and the most significant in the bottom row, so that the mask is read top-to-bottom to determine the capitalization of character from left to right.
- Capitalization mask E2E ASR system are built by tokenizing all training data transcription using the capitalization masks. This system is trained to predict these tokens. At the prediction time , the capitalization are converted back to mixed case using a denormalization step before being presented to the user or used for error evaluation.
  
### Results
#### ASR result on General voice search test set are 
| Model | WER | OWER | CER |
| :--- | :--- | :--- | :---: |
| Lowercase | 6.4 | 1.8 | 33.0 |
| Mixed-Case | 6.5 | 2.2 | 13.4 |
| Cap Masks | 6.5 | 2.2 | 13.1 |

#### ASR results on Contextual Test Sets
**Table: WER on Context Test Sets.**
| Test Set | Model | No Context | Context | Rel. |
|---|---|---|---|---|
| **Contacts** | Lowercase | 16.2 | 5.7 | -65% |
| | Mixed-Case | 16.5 | 5.5 | -67% |
| | Cap. Masks | 16.3 | 5.5 | -66% |
| **Apps** | Lowercase | 6.2 | 2.7 | -56% |
| | Mixed-Case | 6.1 | 3.2 | -48% |
| | Cap. Masks | 6.5 | 2.8 | -61% |

**Table: WER on Anti-Context Test Sets.**
| Test Set | Model | No Context | Context | Rel. |
|---|---|---|---|---|
| **Contacts** | Lowercase | 6.4% | 6.6% | 3% |
| | Mixed-Case | 6.5% | 6.6% | 2% |
| | Cap. Masks | 6.5% | 6.8% | 5% |
| **Apps** | Lowercase | 6.4% | 6.4% | 0% |
| | Mixed-Case | 6.5% | 6.5% | 0% |
| | Cap. Masks | 6.5% | 6.5% | 0% |

---

# Date: 26 Sept 2025
## Paper 12: RNN Approaches to Text Normalization
- Authors: Richard Sproat, Navdeep Jaitly
- Some RNNs produce very good results when measured in terms of overall accuracy, but they produce errors that would make them risky to use in a real applicaation, since in the errorful case, the normalization would convey completely the wrong message.
- But with a pure RNN approach, we have not thus far succeeded in avoiding the above-mentioned risky errors, and it is an open question whether such can be avoided by such a solution.
- In this paper they tried two type of neural models on a text normalization problem.
   - The first is a neural equivalent of a source channel model that uses a seq2seq LSTM that has been sucessfully applied to the grapheme-to-phoneme conversion, along with a standard LSTM language model architecutre.
   - The seconf treats the entire problem such as a seq2seq task, using the same architecture that has been used for speech to text conversion problem.
- This dataset consist of 1.1 billion words of English text and 290 million words of russian text from Wikipedia.

### Text Normalization using LSTMs
- In this approach depends on the observation that text normalization can be broken down into two subproblems. For any token
   - What are the possible normalizations of that token.
   - Which one is appropriate to the given context.
- The first component is a string-to-string transduction problem. Furthermore, since WFSTs can be used to handle most or all of the needed transductions, the relation between the input and output string is regular, so that complex network architecture involving, say, stacks should not be needed. For the input, the string must be in term of characters, since for a string like 123, one needs to see the individual digits in the sequence to know how to read it. Similarly it helps to see the individual character for a possibly OOV word such as a snarky to classify it as a token to be left alone.
- The second components is effectively a language modeling problem, the appropriate level of representation there is words.
- Therefore the output of the first component to the in terms of words.
#### LSTM architecture
- Trained two LSTM models
   - For channel: This LSTM model learn from a sequence of charcters to one or more wors token of output.
      - For most input token this will involve deciding to leave it alone, that is to map it to self or in the case of punctuation to map it to sil, corresponding to silence.
      - For other token it must decide to verbalize it in a variety of diffferent way.
      - used an bidrectional seq2seq model. One with two forward and two backward hidden layer(Shallow model); and one with three forward and then three backward hidden layers (Deep model).
   - For Language: The LSTM system reads the word either from the input, if mapped to self or else from the output if mapped from anything else.
      - This follows the standard RNN language model architecture with an input output layer consisting of |*V*| to 100000 a dimensionality-reduction projection layer, a hidden LSTM layer with a feedback loop and a hierarchical softmax output layer.
      - During training the LSTM learns to predict the next word given the current word but the feedback loop allows the model to build up a history of arbitary length.
- The both models are trained separately.
#### Decoding
- At decoding time we need to combine the outputs of the channel and language model.
- This model is done as follows
   - Each postion in the output of the channel model, we prune the predicted output symbols.
   - If one hypothesis has a very high probablity, they eliminated all other prediction at that position: in practice thid happens in most cases sincr the channel modle os typically very sure of itself at most of the input postions.
   - Also pruned all hypotheses with a low probability (default 0.05). and kept all the *n* best hypotheses at each output position. For these experiment n=5 is kept. 
- All the resulting pruned vectors to populate the corresponding postions in th einput to the LM with the channel probablity is multiplied by the LM probability times as LM weighting factor.
- This method of combining the channel and LM probablities can thought of as a poor-man's equivalent of the composition of a channel and LM weight finite state transducer. The main difference is that there is no straightforward way to represent an arbitary lattice in an LSTM.

#### Results
- The first point observe is that the overall performance is good: the accuracy is about 99% for English and 98% Russian. But nearly all of this can be attributed to the model predicting self for most input tokens and sil for punctuation tokens.
- The performance start to break down, with the lowest perfromance predictably beingfound for cases such as TIME that are not very common in these data.
- In the maximum cases the deep model performed better than shallow model except some cases like money.
- These are entirely due to the channel model: there is nothing ill-informed about hte sequences produced, they just happen to be wrong given the input.
- One way to see this is to compute the oracle accracy, the proportion of the time that the correct answer is in the pseudo-lattice produced by the channel. For the English deep model the oracle accuracy is 0.998. Since the overall accuracy meaning 0.993, that means that 2/7 or about 29% of the error can be attributed to the channel mode not giving the LM a choice of selecting the right model.
  
### Attention-based RNN sequence-to-sequence models
- It models the entire problem as an seq2seq problem. Mapped whole task as an whole task where we map a sequence of input characters to a sequence of the output words.
- They used an tensorflow model with an attention mechanism. Attention models are particularly good for seq2seq problems since they are able to continiously update the decoder with information about the state of the encoder and thus attend better to the relation between the input and output sequences.
- To solve this problem they took a different approach and placed each token in a window of 3 words to the left and 3 word to the right, marking the to-be normalizrd token with a distinctive begin and end tag.
- Specifically they used a 4 layer bidirectional LSTM reader that reads input charcters and a layer of 256 attentional units and 2-layer decoder that produces word sequences.

#### Results
- The performance is mostly better than the previous model. This suggest in turn that modeling the problem as a pure seq2seq transduction is indeed viable as an alternative to the source-channel approach thye had taken previously.
- For English 90.6% of the case not found in the training data were correctly produced (compared to 99.8% of the seen case), and for russian 86.7% of the unseen cases were correct (versus 99.4% of the seen cases).

#### Results on "reasonable" -sized datasets
- the system is again retrained from scratch with 11.4 million tokens of english  and 11.9 million token of Russian.
- The test data overlapped with the training data is 96.9% of the tokens for english and 95.5% fot Russian, with  the accuracy of the non overlapped token being 95.0% for English and 93.5% of Russian.

### Finite-state Filters
- The attention based seq2seq model can produce extremly high accuracies but still prone to occasionally producing output that is completely misleading given the input.
- For this a FST is constucted to guide the decoding. A FST that maps from expressions of the form `number` `measure_abbreviation` to a cardinal or decimal number and the possible verbalization of the measure abbreviation.
- FST implements an overgenrating grammar that includes the correct verbalization, but allow other verbalization as well.
- They constructed a Thrax grammar to cover the measure and money expression, two classes where the RNN is prone to produce silly reading.

## Paper 13: Neural Text Normalization with Subword Units
- Author: Courtney Mansfield, Ming Sun, Yuzong Liu, Ankur Gandhe, Bjorn Hoffmeister
- Parallel writtern/speech formatted text from Sproat and Jaitly
- Non-standard words (NSWs) include expressions such as time, date, abbreviations and letter sequence are commonly appear in written texts such as website, books and movie scripts.
- ASR normalizes the training corpus before building its language model. Among many normalizing the written form text to its spoken form is difficult due to the following bottlenecks:
   - Lack of supervison: there is no incentive for people to produce spoken form text. Thus it is hard to obtain a supervised dataset for training machine learning model.
   - Ambiguity: For written text, a change in context may require a different normalization.

### Model
#### Baseline Model
- Implemented a seq2seq model trained on window-based data.
- This model architecute uses attention to allign the output token with input charcterstics.
- The encoder takes character sequence as input. Otherwise the sequence of numbers or dates are hard to interpret.
- On the output side, they belived that various granularities such as character, word and word fragments can be suitable. a word level decoder is used.
- A window-based seq2seq model, although able to attend well to a central piece os text, is not practical for applying over a whole sentence.
- To extend the model to full sentence, they broke the source sentences into segments. Then apply the model to one segment after another and concatenate their output token to produce full sentences.
- An second baseline, a seq2seq model is trained with full sentence data. It doesnot require any preprocessing step and directly translates a sentence to its spoken form.
- Again the encoder works at the character level while the decoder output sequence of words while attention is used to align the input and output sequences.

#### Proposed Model
- Thesre are several issue wirh the baseline model. Although there is no OOV problem on the input side since it is modeled as a sequence of characters, the decoder has an OOB issue- it cannt be modeled every possible token.
- The window based seq2seq adopts a special output token `self` that significantly reduces the output vocabulary size.
- This model has shown food results in the open-vocabulary speech recognition and machine translation tasks.
- Subwords capture frequently co-occuring character combinations.
- An Extreme case of the subword model is a character model. Compared with only characters, it is belived segmenting input and output into subwords eases a seq2seq model's burden of modeling long distance dependencies.
##### Linguistic Feature
- They also explored the linguistic feature like Casing, POS, and positional feature they are inexpensive to compute but add meaning to the NSW normalization

### Result
**Table: Comparison of models on test set.**
| Model | SER (%) | WER (%) | BLEU | Params (M) | Train Time (hours) | Latency (ms/sent) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Identity | 99.39 | 32.70 | 51.74 | N/A | **0** | N/A |
| Window-based | 12.74 | 3.75 | 94.55 | 10 | 3.9 | 238 |
| Sentence-based | 48.67 | 9.26 | 82.28 | 55 | 8.0 | 159 |
| Subword | **3.31** | **0.91** | **98.79** | 12 | 10.0 | **88** |
| Subword + Feat. w/o label | 2.77 | 0.78 | 98.98 | 12 | 13.5 | 89 |
| Subword + Feat. w/o casing | 0.96 | 0.23 | 99.66 | 12 | 12.8 | **88** |
| Subword + Feat. w/o POS | 0.79 | 0.18 | 99.71 | 12 | 10.4 | **88** |
| Subword + Feat. w/o position | 0.80 | **0.17** | **99.73** | 12 | 13.0 | **88** |
| Subword + All Feat. | **0.78** | **0.17** | **99.73** | 12 | 15.4 | 89 |


---
# Date: 29 Sept 2025
## Paper 14: Punctuation Prediction for Streaming On-Device Speech Recognition
- Authors: Zhikai Zhou, Tian Tan, Yamnin Qian
- Dataset: 3000 hours of in-house chinese spoken utterance with both transcripts and punctuation with three test set Indoor, meeting and Mobile
- Punctuation prediction, also known as punctuation restoration is defined as a sequence tagging task.
- Many works for punctuation prediction have been proposed previously, which can be categorized based on modality: speech modal, text modal, and multimodal containing both.
- Punctuation prediction is treated as a downstream task of unsupervised language models or as an additional task with a pretraining task like the replaced token detection.
- For multi-modality inputs, fundamental frequency and energy are the key features extracted from the speech data.
- Most works treated punctuation prediction as a post-processing task of ASR output.
- The mismatch is usually ignored between the vanilla inputs in the training stage and the ASR hypotheses with errors in the testing stage.
- In this paper, a joint punctuation ASR model is proposed to minimize the number of the additional parameters for the accurate on device punctuation prediction.

### Methods for Punctuation Prediction
#### Two-pass modeling
- Most ASR works consider punctuation prediction as a post processing task for ASR, taking raw text sequence as input and predicting punctuation for each token.
- The language model takes transcripts as input and predicts the punctuation marks for each position.
- In the end, the transcrips and the punctuation are combined to get the final result.

#### One pass modeling
##### Direct ASR Modeling
- A straightforward method is to utilize transcripts with punctuation mark to train the ASR model directly outputs the final results with punctuation marks.

##### Joint Punctuation-ASR Modeling
- The proposed joint model with two series of output for ASR tokens and punctuation marks.
- Also, the model is trained using the multi-task learning framework.
- The parameter of this model can be directly transferred from an existed ASR Model.
- As only the decoder is utilized for actual token prediction in the trigger attention mechanism, the joint model is trained using the following multi-task loss without the CTC loss.
- For the joint punctuation ASR models, method for the joint modeling for punctuation and ASR Tasks, the decoder takes previous tokens and hidden representations as inputs, then the features from different decoder layers are exploited using a linear projection layer for punctuation prediction.
- The method not only takes the previous token as input but previous punctuation marks are also taken.
- The embedding of both tokens and punctuation marks are summed as the input of the ASR decoder.
- In the decoding phase, they conducted beam search or Pure ASR results and take argmax result on punctuation.

##### Teacher-forcing Decoding Scheme
- An important problem for evaluation one-pass pipeline punctuation prediction is that the error from ASR and punctuation prediction are combined.
- Weather the ASR or the punctuation prediction is wrong cannot be determined from any punctuation-related error due to the presence of insertion and deletion errors. Therefore the results of the one-pass models, F1 score cannot be directly calculated.
- For the auto-regressive decoder, the idea of the teacher forcing training is refereed to evaluate the punctuation prediction
- For the one-pass models, the posterior of punctuation marks at time step is calculated. 

### Results
#### Two-pass pipeline with Punctuation Model
**Table: Performance Comparison of the Two-Pass Strategy with Punctuation Models**
| Model | #params | Indoor $\text{TER}/F_1$ | Mobile $\text{TER}/F_1$ | Meeting $\text{TER}/F_1$ |
| :---: | :-----: | :--------------------: | :---------------------: | :----------------------: |
| Trans-2L | 9.88M | 15.93/86.80 | 27.94/70.69 | 28.89/72.64 |
| Trans-4L | 16.19M | 15.88/87.28 | 27.84/71.48 | 28.76/74.09 |
| Trans-6L | 22.49M | 15.72/87.59 | 27.73/71.79 | 28.64/74.15 |
| ASR | 72.6M | 13.49 (CER) | 24.89 (CER) | 25.91 (CER) |

#### One-Pass ASR-Punctuation Models
**Table: Performance comparison of different strategies for both ASR and punctuation prediction. ASR+Trans-6L: The two-pass pipeline using punctuation language models. ASR with punc: The one-pass direct ASR modeling on transcripts with punctuation. Joint Model utilizes feature from which output of the decoder layer: x1: 1st, x2: 3rd, x3: Last, x4: Sum of all, y: Last, but feed punctuation result to the input.**
| Model | $\alpha$ | #Ext par. | Indoor $\text{TER/CER}/F_1$ | Mobile $\text{TER/CER}/F_1$ | Meeting $\text{TER/CER}/F_1$ | Average $\text{TER/CER}/F_1$ |
| :---: | :---: | :-------: | :--------------------------: | :--------------------------: | :---------------------------: | :--------------------------: |
| ASR + Trans-6L | - | 22.49M | 15.72/13.49/87.59 | 27.73/24.89/71.79 | 28.64/25.91/74.15 | 24.03/21.43/77.84 |
| ASR with Punc | - | 11.3K | 15.49/14.45/**92.02** | 31.73/28.87/71.66 | 31.88/29.95/78.06 | 26.37/24.42/80.58 |
| Joint Model -x3 | 1.0 | 2.0K | 14.62/**13.19**/91.01 | 24.27/21.39/72.38 | **27.76**/**25.33**/78.82 | 22.22/19.97/80.74 |
| Joint Model -x3 | 2.0 | 2.0K | **14.51**/**13.20**/91.45 | **23.66**/**20.60**/71.70 | 28.46/26.15/78.29 | **22.21**/**19.98**/**80.48** |
| Joint Model -x3 | 5.0 | 2.0K | 14.53/13.36/92.00 | 24.48/21.59/**72.17** | 28.77/26.53/**79.11** | 22.59/20.49/**81.09** |
| Joint Model -x1 | 2.0 | 2.0K | 39.51/17.54/50.51 | 57.92/35.58/35.21 | 46.35/37.74/53.51 | 47.93/30.29/46.41 |
| Joint Model -x2 | 2.0 | 2.0K | 20.10/13.68/79.84 | 35.44/25.99/58.57 | 34.68/27.77/69.74 | 30.07/22.48/69.38 |
| Joint Model -x3 | 2.0 | 2.0K | **14.51**/**13.20**/91.45 | **23.66**/**20.60**/71.70 | 28.46/26.15/78.29 | **22.21**/**19.98**/**80.48** |
| Joint Model -x4 | 2.0 | 2.0K | 14.61/13.23/91.17 | 25.31/22.40/70.91 | **28.29**/**25.83**/77.72 | 22.74/20.49/79.93 |
| Joint Model -y | 2.0 | 4.0K | 14.56/13.24/91.20 | 24.82/21.95/**72.30** | 29.23/27.01/**78.46** | 22.87/20.73/**80.65** |

## Paper 15: Streaming Punctuation For Long-Form Dictation with Transformers
- Authors: Piyush Behre, Sharman Tan, Padma Varadharajan and Shuangyu Chang
- Dataset: Uses publicly availabe dataset from the various domains

| Dataset | Distribution |
| :--- | :---: |
| **OpenWebText** | **52.8%** |
| **Stack Exchange** | **31.5%** |
| OpenSubtitles2016 | 7.6% |
| MAEC | 6.7% |
| NPR Podcast | 1.4% |

**Table: Data distribution by number of words per dataset**

- In this hybrid approach ASR generates pucntuation with two systems working together. Firstly decoder generates text segmentation and passes them to Display Post Processor (DPP).
- Works well for Single shot uses like voice assitant and voice search but fails for Long-form dictation.
- Dictation session typically comprises many spoken form text segments generated by the decoder. Decoder features such as speaker pause duration determine the segment boundaries.
- The punctuation model in DPP then punctuates each of those segments.
- Without cross-segment look-ahead or the ability to correct previously finalized results, the punctuation model fucntions within the boundaries of each provided text segments.
- Introduced a novel streaming punctuation approach to punctuate and repunctuate ASR outputs, demonstrated streaming punctuation's robustness to model architecture choices through experiments and achived not only gain in punctuation quality but also significant downstrams BLEU score gains or Machine translation for a set of languages.

### Proposed Method
#### Punctuation Model
- Framed the punctuation prediction as a sequence tagging problem.
- First tokenize the input segment as byte-pair encoding(BPE) tokens and pass this through a transfomer encoder.
- Next a punctuation token classification head, consisting of a dropout layer and a fully connected layer, generates token level punctuation tags.
- Finally converted the token level tags to word level and generated punctuated text by appending each specified punctuation symbol to the corresponding word in the input segment.

#### Streaming Decoder for Punctuation
- Hybrid ASR system often define segmentation boundaries using silence thresholds. However, for Human2machine scenarios like dictation, pauses do not necessarily indicates ideal segmentation boundaries for the ASR system.
- For dictation users, this system would produce a lot of over segmentation. To solve this issue, must incorporate the right context across segment boundaries.
- They proposed solution is a streaming punctuation system. the key is to emit complete sentence only after detecting the beginning of a new sentence.
- At each step they punctuated text within a dynamic decoding window. This window consists of a buffer for which the system hasnot yet detected a sentence boundry and the new incoming segment.
- When at least one sentence boundary is detected within the dynamic window, all complete sentence are omitted and reserve the remaining text as the new buffer.
- This strategy discards the orginal decoder boundary and decides the sentence boundary purely on the linguistic features.

### Results
#### Punctuation, Segmentation and Downstream task Results
**Table: Punctuation results**
| Test Set | Model | PERIOD | Q-MARK | COMMA | OVERALL | $F_1$-Gain |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | | **P** R $F_1$ | **P** R $F_1$ | **P** R $F_1$ | **P** R $F_1$ | |
| Dict-100 | BL-LSTM | 64 71 67 | 47 88 61 | 62 52 57 | 63 61 61 | - |
| | ST-LSTM | 77 63 69 | 67 71 69 | 60 52 56 | 68 57 62 | 0.6% |
| | BL-Transformer | 69 76 72 | 50 88 64 | 68 52 59 | 68 63 65 | - |
| | ST-Transformer | 81 71 76 | 82 82 82 | 69 51 59 | 74 60 67 | 2.9% |
| MAEC | BL-LSTM | 68 79 73 | 46 44 45 | 63 50 56 | 65 63 64 | - |
| | ST-LSTM | 77 70 73 | 65 45 54 | 60 51 55 | 68 60 64 | 0.0% |
| | BL-Transformer | 71 80 75 | 50 50 50 | 65 49 56 | 67 63 65 | - |
| | ST-Transformer | 80 78 79 | 69 46 56 | 65 48 55 | 72 62 66 | 2.4% |
| EP-100 | BL-LSTM | 56 71 63 | 64 62 63 | 55 47 51 | 56 58 56 | - |
| | ST-LSTM | 70 62 66 | 69 55 61 | 57 49 53 | 63 55 59 | 4.2% |
| | BL-Transformer | 58 76 66 | 58 70 64 | 57 49 53 | 57 61 59 | - |
| | ST-Transformer | 70 71 71 | 76 70 73 | 59 51 55 | 64 60 62 | 5.8% |
| NPR-76 | BL-LSTM | 72 71 72 | 71 66 69 | 65 58 61 | 69 65 67 | - |
| | ST-LSTM | 82 71 76 | 76 69 73 | 65 59 62 | 74 66 70 | 4.0% |
| | BL-Transformer | 76 77 76 | 76 70 73 | 68 60 64 | 72 69 71 | - |
| | ST-Transformer | 87 79 83 | 81 75 78 | 70 61 65 | 79 71 75 | 6.0% |


**Table: Segmentation Results**
| Test Set | Model | Segmentation | $F_1$-gain | $F_{0.5}$ | $F_{0.5}$-gain |
| :---: | :---: | :---: | :---: | :---: | :---: |
| | | **P** R $F_1$ | | | |
| Dict-100 | BL-LSTM | 62 68 65 | - | 63 | - |
| | ST-LSTM | 74 60 66 | 1.5% | 71 | 12.0% |
| | BL-Transformer | 66 74 70 | - | 67 | - |
| | ST-Transformer | 79 69 73 | 4.3% | 77 | 13.8% |
| MAEC | BL-LSTM | 66 76 71 | - | 68 | - |
| | ST-LSTM | 76 68 72 | 1.4% | 74 | 9.5% |
| | BL-Transformer | 69 77 73 | - | 70 | - |
| | ST-Transformer | 79 75 77 | 5.5% | 78 | 10.9% |
| EP-100 | BL-LSTM | 53 67 59 | - | 55 | - |
| | ST-LSTM | 66 58 62 | 5.1% | 64 | 16.1% |
| | BL-Transformer | 54 72 62 | - | 57 | - |
| | ST-Transformer | 67 68 68 | 9.7% | 67 | 18.2% |
| NPR-76 | BL-LSTM | 71 70 70 | - | 71 | - |
| | ST-LSTM | 81 70 75 | 7.1% | 79 | 10.9% |
| | BL-Transformer | 74 75 75 | - | 74 | - |
| | ST-Transformer | 85 79 81 | 8.0% | 84 | 12.5% |

**Table: Translation BLEU Results: English audio recognized, punctuated, and translated to 7 languages**
| Language | Model | BLEU | Gain |
| :---: | :---: | :---: | :---: |
| de | BL-LSTM | 36.0 | |
| | ST-LSTM | 36.6 | +0.6 |
| | BL-Transformer | 36.4 | |
| | ST-Transformer | 37.5 | +1.1 |
| fr | BL-LSTM | 41.0 | |
| | ST-LSTM | 40.6 | -0.4 |
| | BL-Transformer | 41.7 | |
| | ST-Transformer | 41.8 | +0.1 |
| el | BL-LSTM | 39.8 | |
| | ST-LSTM | 40.8 | +1.0 |
| | BL-Transformer | 40.3 | |
| | ST-Transformer | 41.7 | +1.4 |
| It | BL-LSTM | 35.2 | |
| | ST-LSTM | 35.5 | +0.3 |
| | BL-Transformer | 35.4 | |
| | ST-Transformer | 35.9 | +0.5 |
| pl | BL-LSTM | 30.2 | |
| | ST-LSTM | 30.9 | +0.7 |
| | BL-Transformer | 31.1 | |
| | ST-Transformer | 31.7 | +0.6 |
| pt | BL-LSTM | 33.2 | |
| | ST-LSTM | 33 | -0.2 |
| | BL-Transformer | 33.7 | |
| | ST-Transformer | 33.9 | +0.2 |
| ro | BL-LSTM | 39.8 | |
| | ST-LSTM | 40.1 | +0.3 |
| | BL-Transformer | 40.5 | |
| | ST-Transformer | 41.2 | +0.7 |

---
# Date: 06 Oct 2025
## Paper 14: DATA2VEC-AQC: SEARCH FOR THE RIGHT TEACHING ASSISTANT IN THE TEACHER-STUDENT TRAINING SETUP
- Authors: Vasista Sai Lodagala, Shreyan Ghosh, S. Umesh
- Dataset:
- Proposed data2vec-aqc, a novel SSL-based pre-training methodology from learning speech representation from low resource unlabeled speech.
- Firstly made data2vec simultaneously solve a MAM-based cross-contrastive task between the student and teacher networks by passing randomly augmented versions of the same audio sample passed through each network.
- Then added quantizer module similar as sampling negatives from the qualtized representations.
- Then introduced a clustering module to the cluster the quantized representation and control the effect of those negatives in the contrastive loss computation, that share the same cluster as the positive.

### Methodology
- Standard data2vec architecture involves a student and teacher network, both of which see raw speech as the input and the teacher's parameter are updated based on an exponential moving avarage of the students coder.
- A simple L<sub>2</sub> loss is computed between the student embedding and the average of the embedding from the top 8 layer of the teacher network as it works better than L<sub>1</sub> loss in speech processing.
- 

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
## ASR perfomance without normalizations
Evaluated ASR perfomance on the Test dataset `dataset_all.json` using different pretrained ASR model like `indicwav2vec-hindi`,`Wav2Vec2-large-xlsr-hindi`,`vakyansh-wav2vec2-hindi-him-4200`, and `wav2vec2-large-xlsr-53` using the code `nonormalization.py` and all the results are stored in `Without Normalization Results` directory.

| Model | WER (%) | CER (%) | Link |
| :---: | :---: | :---: | :---: |
| indicwav2vec-hindi | 21.38 | 7.79 | https://github.com/diptiman-mohanta/Text-Normalization-for-Automatic-Speech-Recognition/tree/main/Without%20Normalization%20Results/ai4bharat |
| Wav2Vec2-large-xlsr-hindi | 68.76 | 28.19 | https://github.com/diptiman-mohanta/Text-Normalization-for-Automatic-Speech-Recognition/tree/main/Without%20Normalization%20Results/Wav2Vec2-large-xlsr-hindi |
| vakyansh-wav2vec2-hindi-him-4200 | 24.64 | 8.94 | |

## ASR performance with normalization
| Normalizer | Model(For Transcription) | WER (%) | CER (%) | Link |
| :---: | :---: | :---: | :---: | :---: |
| IndicNLP | indicwav2vec-hindi | 20.52 | 7.37 | |
| | Wav2Vec2-large-xlsr-hindi | 69.3 | 28.43 | |
| | vakyansh-wav2vec2-hindi-him-4200 | 23.81 | 8.51 | |
| Whisper | indicwav2vec-hindi | 8.66 | 3.93 | |
| | Wav2Vec2-large-xlsr-hindi | 41.84 | 21.19 | |
| | vakyansh-wav2vec2-hindi-him-4200 | 9.82 | 4.63 | |
