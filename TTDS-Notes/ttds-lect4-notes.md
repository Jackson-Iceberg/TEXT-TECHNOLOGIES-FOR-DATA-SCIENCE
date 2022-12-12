# lect4

## **Preprocessing**

- Tokenisation
- Stopping
- Normalisation
    - Stemming
    

### **Preprocessing:** identify the optimal form of the term to be indexed to achieve the best retrieval performance.

### **Tokenisation**

- Sentence → tokenization (splitting) → tokens
- A token is an instance of a sequence of characters

**Tokenisation Problems:**

- 语义不详
- 复合词 应该算一个token还是两个
- 数字，URL，Social Media如何被tokenization

**Tokenisation: common practice**

- Just split at non-letter characters.
- Add special cases if required.
- Some applications have special setup
    - Social media: hashtags/mentions handled differently
    - URLs: no split, split at domain only, remove entirely!
    - Medical: protein & diseases names

### Stopping(stop words removal)

- What is stop words: the most common words in collection: the,a,is,he,she,I,him,her…. [STOP_WORDS_LIST](http://members.unine.ch/jacques.savoy/clef/index.html)
- Stop words influence on sentence structure and less influence on topic.
- Stop words list can be created by ourself:
1. Sort all terms in a collection by frequency.
2. Manually select the possible stop words from Top N terms.

### Normalisation

- Make words with different surface forms look the same.
- Case folding and equivalents to achieve it.
    - “A” & “a” are different string for computers.
    - CAR,Car,car → car OR CAR
    - English: China; 中文：中国
    - multi-disciplinary → multidisplinary ← multi disciplinary
- The most important criteria:
• Be consistent between documents & queries
• Try to follow users’ most common behaviour

### Stemming-词干提取&词干总结

- Play: played, playing,player.
- Many morphological variations of words
• inflectional (plurals, tenses)
• derivational (making verbs nouns etc.)
- Stemmers attempt to reduce morphological variations of words to a common stem.(usually involves removing suffixes (in English)

**Stemming methods:**

- **Dictionary-based:** uses lists of related words
- **Algorithmic**: uses program to determine related words： Porter Stemmer, remove “s” ending assuming plural.
- **Stemmed words are misspelled**

**Porter Stemmer(是一种常用于IR的词干提取方法）**

- **a process for removing the commoner morphological and inflexional endings from words in English. Its main use is as part of a term normalisation process that is usually done when setting up Information Retrieval systems.**

### Pre-processing: Common practice

- Tokenisation: split at non-letter characters
    - Basic regular expression
     → process \w and neglect anything else
    - For tweets, you might want to keep “#” and “@”
- Remove stop words
    - find a common list, and filter these words out
- Apply case folding
    - One command in Perl or Python: lc($string)
- Apply Porter stemmer

### Pre-processing-Limitations

- **Irregular verbs:**
    - saw → see
    - went → go
- **Different spellings**
    - colour vs. color
    - tokenisation vs. tokenization
    - Television vs. TV
- **Synonyms**
    - car vs. vehicle
    - UK vs. Britain

**Potential solutions:**

![lect4-img1](https://github.com/Jackson-Iceberg/TTDS-Notes/blob/main/images/lect4-img1.png)
