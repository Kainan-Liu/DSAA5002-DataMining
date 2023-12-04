# DSAA5002-DataMining
This is the DSAA5002 final project--- Data Mining and Knowledge Discovery in Data Science

![image](https://github.com/Kainan-Liu/DSAA5002-DataMining/assets/146005327/939fc4f0-ca33-49e0-a52d-05e62acb242d)

**This project consists of two tasks**

## Task 1 Data Preprocessing and Analysis
**Background:** Assuming you are a sentiment analyst at a securities firm, your task is to assess the impact of each news article on the A-share listed companies explicitly mentioned. For instance, on October 14, 2022, the China Securities Journal(中国证券报) reported the following:

![image-20231109134533359](https://github.com/Kainan-Liu/DSAA5002-DataMining/assets/146005327/361c6e5e-4942-4835-b248-e8cf3299bf70)

This news explicitly mentions three companies: 贵州茅台(600519.SH), 今世缘(603369.SH), and 水井坊(600779.SH), and the impact appears to be positive. "Positive" indicates that this news appears to positively affect the company's stock price

### SubTask1: Data Preprocessing - Noise Removal

**Input**: News.xlsx, which includes **1,037,035** pieces of news. Please download it from: https://docs.google.com/spreadsheets/d/1VAzteetSSc9WOCne_u6-5oFt_6rIMR5E/edit?usp=share_link&ouid=112799952654350672254&rtpof=true&sd=true

**Description**: This is an open-ended question. Given the input, we consider any news that does not mention any of the China A-share listed companies to be noise. Task is to remove rows in the data table News.xlsx that do not mention any of the China A-share listed companies(as provided in A_share_list.json).

### SubTask2: Data Analysis - Sentiment Analysis

**Input**: The news data was cleaned through Question 1

**Description**: Building on Question 1, we assume you have obtained a clean dataset where each news mentions at least one A-share listed company. In this question, your objective is to determine the sentiment polarity of each news text. This task can be treated as a binary classification problem, with Class 0 indicating "negative" and Class 1 indicating "positive". You can refer to the submission_excel_sample/sample_Task1.xlsx file for more information

## Task 2 Application of Knowledge Graph 

Background: In addition to explicitly mentioning listed companies, each news article may also implicitly impact the other companies, either positively or negatively. For instance, on October 14, 2022, the China Securities Journal(中国证券报) reported the following:

> 2022-10-14 截至10 月13 日，包括贵州茅台、今世缘、水井坊等多家酒企披露了前三季度业绩预告或经营数据公告。从目前已披露的数据来看，前三季度酒企业绩表现稳定。

This news explicitly mentions three companies:  贵州茅台(600519.SH),  今世缘(603369.SH), and 水井坊(600779.SH), and the impact appears to be positive. However, other companies such as  五粮液  (000858.SZ),  洋河股份  (002304.SZ),  泸州老窖 (000568.SZ), and  山西汾酒(600809.SH) might also be positively affected, as they belong to the same industry as  贵州茅台(600519.SH).  Conversely,  this  news  might  have  a  negative  impact  on  贵绳股份(600992.SH)  and  宁德时代(300750.SZ), as  贵绳股份  has a dispute with  贵州茅台, and  宁德时代  competes with  贵州茅台.   
 
In the above analysis, expressions like "belong to the same industry," "have a dispute," and "compete" can be considered as forms of knowledge. The most well-known data structure for representing knowledge is a knowledge graph, where nodes represent entities, and edges represent relationships. Each relationship connects two entities and can be represented using a triple (S, P, O) = (Subject, Predicate, Object). For example, "贵州茅台(600519.SH) and  贵绳股份(603369.SH) have a dispute" can be represented as a triple (600519.SH, "dispute", 603369.SH), and "宁德时代(300750.SZ) competes with 贵州茅台(600519.SH)" can be represented as a triple (300750.SZ, "compete", 600519.SH).   
 
Fortunately, a research team has studied the relationships between A-share listed companies and provided the data in the KnowledgeGraph folder. In the following questions, you are required to use all the data from the KnowledgeGraph folder.

### SubTask1: Constructing a Knowledge Graph 
In the knowledge graph you build,  the  node  type  is  "company,"  and  there  are  six  types  of  edges:  "compete,"  "cooperate,"  "dispute,"  "invest,"  "same_industry," and "supply." Edges can be directed, meaning from S to P, or undirected (bidirectional)

![image](https://github.com/Kainan-Liu/DSAA5002-DataMining/assets/146005327/a556d51d-a6a6-408b-b75b-89af75117677)

### SubTask2: Knowledge-Driven Financial Analysis
Identify ALL implicit companies corresponding to each company  of Explicit_Company in your own Task1.xlsx file. Categorize them into Implicit Positive Companies and Implicit Negative Companies. 
![image](https://github.com/Kainan-Liu/DSAA5002-DataMining/assets/146005327/b3f0ffd7-716b-4aa5-a87f-c60d8cdf3f3c)

