# Lect2

**IR-Information Retrieval**

- **Given Query Q, find relevant documents D**

IR的两个难点重点

- Effectiveness
    1. 需要找到正确的相关文本信息
    2. 不同于传统数据库如SQL
- Efficiency
    1. 需要快速检索，检索时间不能过长
    2. 数据量极大
    3. 正常检索速度为：thousands queries per second (Google, 99,000)
    4. 数据库内容数据经常会改变，需要及时更新
    5. 相较于NLP，IR检索速度更快

IR主要组件

- Documents 文件
- Queries
- Relevant Documents 相关文件

Documents特点

- Unstructured
- unique ID

Queries特点

- Free text to express user’s information need
- Same information can be described by mulptiple queries. 同义词
- Same query can represent multiple different meaning e.g: Apple

Challenge in relevance

- No clear semantics, contrast. 语义/意图不明确
- Inherent ambiguity of language 语言本身的歧义
    - Synonymy: “Edinburgh festival” = “ The fringe”
    - Polysemy: “Apple”, “Jaguar”
- Relevance highly subjective 相似词的个人主观性

**IR的Database和normal Database区别**

![lect2-img1](https://github.com/Jackson-Iceberg/TEXT-TECHNOLOGIES-FOR-DATA-SCIENCE/blob/main/images/lect2-img1.png)

IR系统

- Indexing Process: offline
    - Get the data into system.
        - Acquire data, store, transform to BOW and “index”
- Search(retrieval) Process: online
    - satisfy users’ requests
        - assist user in formulating query
        - retrieve a set of results
        - help user browse/re-formulate
        - according user’s actions, adjust retrieval model.
    

![lect2-img2](https://github.com/Jackson-Iceberg/TEXT-TECHNOLOGIES-FOR-DATA-SCIENCE/blob/main/images/lect2-img2.png)

![lect2-img3](https://github.com/Jackson-Iceberg/TEXT-TECHNOLOGIES-FOR-DATA-SCIENCE/blob/main/images/lect2-img3.png)
