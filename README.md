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

#### Solution
Use Chinese-Bert based model pretrained on Chinese Financial News

https://huggingface.co/hw2942/bert-base-chinese-finetuning-financial-news-sentiment-v2

![image-20231109135613887](https://github.com/Kainan-Liu/DSAA5002-DataMining/assets/146005327/33703a42-c36c-4bba-97df-9c67f5e146dd)



