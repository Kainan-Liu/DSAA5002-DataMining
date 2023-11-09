# DSAA 5002 - Data Mining and Knowledge Discovery in Data Science

<img src="https://www.netsuite.com/portal/assets/img/business-articles/data-warehouse/social-data-mining.jpg?v2" alt="What Is Data Mining? How It Works, Techniques & Examples | NetSuite" style="zoom:67%;" />

**This project consists of two tasks**

## Task 1 Data Preprocessing and Analysis

**Background:** Assuming you are a sentiment analyst at a securities firm, your task is to assess the impact of each news article on the A-share listed companies explicitly mentioned. For instance, on October 14, 2022, the China Securities Journal(中国证券报) reported the following:

![image-20231109134533359](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20231109134533359.png)

This news explicitly mentions three companies: 贵州茅台(600519.SH), 今世缘(603369.SH), and 水井坊(600779.SH), and the impact appears to be positive. "Positive" indicates that this news appears to positively affect the company's stock price

### SubTask1: Data Preprocessing - Noise Removal

**Input**: News.xlsx, which includes **1,037,035** pieces of news. Please download it from: https://docs.google.com/spreadsheets/d/1VAzteetSSc9WOCne_u6-5oFt_6rIMR5E/edit?usp=share_link&ouid=112799952654350672254&rtpof=true&sd=true

**Description**: This is an open-ended question. Given the input, we consider any news that does not mention any of the China A-share listed companies to be noise. Task is to remove rows in the data table News.xlsx that do not mention any of the China A-share listed companies(as provided in A_share_list.json).

### SubTask2: Data Analysis - Text Knowledge Mining

**Input**: The news data was cleaned through Question 1

**Description**: Building on Question 1, we assume you have obtained a clean dataset where each news mentions at least one A-share listed company. In this question, your objective is to determine the sentiment polarity of each news text. This task can be treated as a binary classification problem, with Class 0 indicating "negative" and Class 1 indicating "positive". You can refer to the submission_excel_sample/sample_Task1.xlsx file for more information

![image-20231109135613887](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20231109135613887.png)
