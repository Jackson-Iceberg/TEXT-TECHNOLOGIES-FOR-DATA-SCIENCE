# Lect5

### Indexing Processing

**IR-Index**

- Data structure for fast finding terms.
- Additional optimisations could be applied.

Document Vectors

- Represent documents as vectors
    - Vector→document,cell→term
    - Values: term frequency or linary (0,1)
    - All documents → collection matrix

**Inverted index**

- an umbrella term for many different kinds of structures that share the same general philosophy.
- If a document gets a high score, this means that the system thinks
that document is a good match for the query, whereas lower numbers mean that the system thinks the document is a poor match for the query. To build a ranked list of results, the documents are sorted by score so that the highest-scoring documents come first.
- an inverted index is organized by index term. The index is inverted
because usually we think of words being a part of documents, but if we invert this idea, documents are associated with words.

Documents→IR(inverted index)

![lect5-img1](https://github.com/Jackson-Iceberg/TTDS-Notes/blob/main/images/lect5-img1.png)

IR+word counts

![lect5-img2](https://github.com/Jackson-Iceberg/TTDS-Notes/blob/main/images/lect5-img2.png)

IR+word positions

![lect5-img3](https://github.com/Jackson-Iceberg/TTDS-Notes/blob/main/images/lect5-img3.png)

- to avoid these are two sentences, on Coincidence to match the two words, so adding the restriation of index OR position, to ensure the words are in the same sentence.防止因为巧合，本来两个不相干的词，处于上下句结构，导致误以为match上了。解决方法如下图，拆分句子storing
- Storing way: feature function values, the values means the score that this sentence gain. Higher score means higher rank.
    
    ![lect5-img4](https://github.com/Jackson-Iceberg/TTDS-Notes/blob/main/images/lect5-img4.png)
    

**Scores**

- “fish” that has postings like [(1:3.6), (3:2.2)], meaning that the total feature value for “fish” in document 1 is 3.6, and in document 3 it is 2.2.
- Presumably the number 3.6 came from taking into account how many times “fish” appeared in the title, in the headings, in large fonts, in bold, and in links to the document. Maybe the document doesn’t contain the word “fish” at all, but instead many names of fish, such as “carp” or “trout. The value 3.6 is then some indicator of how much this document is about fish.

**Boolean Search**

- Boolean: exist/not exist (1 means exist,0 means not exist)
- Logical operators(AND,OR,NOT)
- Build a Term-Document Incidence Matrix
    - Rows are terms
    - Columns are documents

**Collection Matrix**

![lect5-img5](https://github.com/Jackson-Iceberg/TTDS-Notes/blob/main/images/lect5-img5.png)

- In big collections, because Zip’s law → 250k terms appears once
- hence Collection matrix is extremely sparse.(mostly 0’s)

**Inverted index: Sparse representation**

- for each term t, we must store a list of all documents that contains t.
- Identifying each by a **docID, a document serial number**.

![lect5-img6](https://github.com/Jackson-Iceberg/TTDS-Notes/blob/main/images/lect5-img6.png)

Inverted Index Construction

![lect5-img7](https://github.com/Jackson-Iceberg/TTDS-Notes/blob/main/images/lect5-img7.png)

![lect5-img8](https://github.com/Jackson-Iceberg/TTDS-Notes/blob/main/images/lect5-img8.png)

![lect5-img9](https://github.com/Jackson-Iceberg/TTDS-Notes/blob/main/images/lect5-img9.png)

![lect5-img10](https://github.com/Jackson-Iceberg/TTDS-Notes/blob/main/images/lect5-img10.png)

![lect5-img11](https://github.com/Jackson-Iceberg/TTDS-Notes/blob/main/images/lect5-img11.png)

Phrase Search & Proximity Index

![lect5-img12](https://github.com/Jackson-Iceberg/TTDS-Notes/blob/main/images/lect5-img12.png)

Query Processing: Proximity & Proximity search: data structure

![lect5-img13](https://github.com/Jackson-Iceberg/TTDS-Notes/blob/main/images/lect5-img13.png)
