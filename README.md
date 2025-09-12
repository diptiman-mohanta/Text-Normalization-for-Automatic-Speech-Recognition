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
- Model Architecture: A A BERT-based encoder with a multi-layer perceptron (MLP) and softmax for token classification. It processes the entire spoken sentence in one pass, assigning tags to each input token.

### Proposed Approach:
- Initial data: To train this model they alighedn the GTN dataset. The GTN dataset consists of
unnormalized (i.e. written form) and normalized (i.e. spoken form) sentence pairs that are aligned on a phrase-level. To get  amonotonic one to one correspondence between each spoken word and corresponding fragments in writtten from the dataset is alligned into more granular level.
- Alignment: All corresponding phrases are extracted to create a parallel corpus for each semiotic class. Used Giza++ for this allignment. To do this alignment they tokenize the data first The spoken text istokenized by word boundary, while the written part is tokenized as follows: 1) All alphabetic sequences are separate tokens, 2) In numeric sequences each character is a separate token. 3) All non-alphanumeric characters are separate tokens. Additionally, we add an underscore symbol to mark the beginning and end of a sequence for future detokenization. For example, "jan 30,2005" is tokenized as "_jan_ _3 0 , 2 0 0 5_". Then they joined thogether the character token in the written part that are aligned to the same spoken input word. If a spoken input word aligns to nothing, we add a "<DELETE>" tag.
- Non-Monotonic alignment: One important restriction of the tagger mode is that spoken and written pairs are assumed to be monotonically aligned. Most of the spoken-written pairs in the dataset satisfy the requirement. Detected the non-monotonic examples and discarded them.
- Tag Vocabulary and Training Dataset: The alignment procedure splits the written sentences into fragme ts and align them one-to-one to be spoken form. Then count the frequencies of such written fragments and include some predefinec number of the most frequent of them in the tag vocabulary.  "<SELF>"and"<DELETE>"tags are also added to the tag vocabularies to signify that an input token should be either copied or deleted. The tagger model sees the whole sentence to make context dependent decisions. All input tokens outside ITN spans are mapped to"<SELF>"tags during data set creation.
- Post Processing: To get the final output applied a simple post processing procedure. specifically substituted  "<SELF>"tokens with input words,remove"<DELETE>" tokens, move tokens that have movements encoded in tags, and, finally, remove spaces between fragments bordered with under score symbols.
