# lect3

### **Laws of Text**

- Zipf’s law
- Benford’s law
- Heap’s law
- Clumping/contagion

### Zipf’s Law

- rank*Pr ≈ constant
- r, rank of term according to frequency. Frequency is high, rank is high.
- Pr, probability of appearance of term.
- **1/rank of word * constant = frequency of the word**
    
![lect3-img1](https://github.com/Jackson-Iceberg/TTDS-Notes/blob/main/images/lect3-img1.png)
    
- rank越靠前，该单词出现的频率越高；Rank*单词频率=常量，作用：可以根据单词的Rank来推断频率，或者通过频率来决定Rank。这样可以帮助排序哪个单词更有可能出现，从而排先后顺序。
- THE出现的频率是1的话，OF出现的频率是1/2，AND出现的频率是1/3，TO是1/4，以此类推。
    
![lect3-img2](https://github.com/Jackson-Iceberg/TTDS-Notes/blob/main/images/lect3-img2.png)
    
- 5555是sauce的rank，frequecny最后查到是29869。181million是THE的frequency

![lect3-img3](https://github.com/Jackson-Iceberg/TTDS-Notes/blob/main/images/lect3-img3.png)

- Frequency of words has hard exponential decay

![lect3-img4](https://github.com/Jackson-Iceberg/TTDS-Notes/blob/main/images/lect3-img4.png)

### Benford’s Law

- First digit of a number follows a Zipf’s like law.
- P(d) = log(1+1/d)
- 

![lect3-img5](https://github.com/Jackson-Iceberg/TTDS-Notes/blob/main/images/lect3-img5.png)

- P(1) = ln(1+1/1)/ln(10) = 0.301.
- P(2) = ln(1+1/2)/ln(10) = 0.185
- 1 = 33.4%
- 2 = 18.5%
- 3 = 12.4%
- 4 = 7.5%
- 5 = 7.1%
- 6 = 6.5%
- 7 = 5.5%
- 8 = 4.9%
- 9 = 4.2%
- 本来按照数学期望 mathematics expectation 每个数字成为第一位都是11.1%,但是实际情况却是 digit 1 出现在第一位的概率为30%，digit 2 出现在第一位的概率为18.5%…..

### Heap’s Law

- While going through documents, the number of new terms noticed will reduce over time.
- Heaps' law means that as more instance text is gathered, there will be diminishing returns in terms of discovery of the new vocabulary from which the distinct terms are drawn.
- For a book/collection, while reading through, record:
• n: number of words read
• v: number of news words (unique words)
- Vocabulary growth: v(n) = k*n^b
- where b<1, 0.4<b<0.7
- 

![lect3-img6](https://github.com/Jackson-Iceberg/TTDS-Notes/blob/main/images/lect3-img6.png)

### **Clumping/Contagion in text**

- Majority of terms appearing only twice appear close to each other.
- 一个词很少单独出现，如果出现了一次，那么距离它下一次出现的距离肯定不算远。
- 所以说词语具有传染性。或者成clumping形态出现。

**补全遗漏的知识&拓展盲区**

**Logarithms:**

- 本质上是为了解决特别大的数。
- log(b)n = p ⇒  b^p = n   比如 log(10)100000 = 5 ⇒ 10^5 = 100000
- 这样能解决过大或过小的数字，使用log能使数值变得更加直观，而不是数零

**Pareto Principle**: 20/80法则，28法则
