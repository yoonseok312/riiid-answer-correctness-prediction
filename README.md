# Riiid AIEd Challenge 2020
## 149th solution (Top 5%, AUC 0.793) of [Riiid! Answer Correctness Prediction](https://www.kaggle.com/c/riiid-test-answer-prediction)

From October 12th, 2020 to January 7th, 2021, AIEd company Riiid! hosted a Kaggle competition that asked participants to create algorithms for "Knowledge Tracing," the modeling of student knowledge over time. The goal was to accurately predict how students will perform on future interactions, based on a [EdNet dataset](https://github.com/riiid/ednet) provided by Riiid.  A total of 3,406 teams/4,412 participants from 100+ countries submitted 64,678 models to explore and tackle AI for education. 

For 2 months, I participated in the competition as a team leader with Harheem Kim, Hung Giang, Sanmaru Um, and Yeseul Gong. We had a lot of limitations: none of us had experience with AI/ML, none of us have particiapated in a Kaggle competition before, and we didn't have any additional hardware support such as GPU machines that a lot of winning teams had. Nevertheless, we were able to grow in a fast pace and secure a Silver medal granted to top 5% of participants with a bit of luck. This repo contains a solution write up for our model, which is an ensemble between a single Light Gradient Boosted Machine model, and a single Encoder-Decoder based Transformer model. Should you have any questions, feel free to contact me at yoonseok@berkeley.edu.

# LightGBM
### Features
* **ts_delta**: the gap between timestamp of current content with previous content of the same user.
* **task_container_id**
* **prior_question_elapsed_time**: in second, rounded
* **prior_question_had_explanation**
* **part**
* **num_tag**: number of tags in that question
* **u_chance**: the average correctness of the user until current time
* **u_attempts**: number of content the user have done
* **u_attempt_c**: number of times the user interacted with the specific content in the past (only counting from >1 interactions due to memory)
* **c_chance**: the average correctness of the question until current time
* **c_attempts**:  number of encounter of that question (all user)
* **u_part_chance**: the average correctness of the user doing the same part as the question
* **u_part_attempts**: number of question of the same part the user have done
* **u_skill_chance**: the average correctness of the user doing the same skill as the question (part < 5: listening, part >= 5: reading)
* **u_skill_attempts**: number of question of the same skill the user have done
* **t_chance**: the average correctness of the user of questions with specific tag until current time
* **t_attempts**: user's number of encounter of that tag
* **total_explained**: number of times explanation was provided to the user until current time (all contents)
* **10_recent_correctness**: user correctness of the most recent questions (up to 10)
* **10_recent_mean_gap**: mean ts_delta of the most recent questions (up to 10)
* **bundle_elapsed**: the mean elapsed time of the bundle, up until the abs time.
* **mean_elapsed**: the mean elapsed time of the user until now. 
* **prev_t1**: tag of the last question
* **prev_cor**: correctness of the last question
* **trueskill_possibility**: possibility of the user 'beating' the question (getting the question correct) based on [trueskill](https://trueskill.org/)
* **mu**: mu value (mean of trueskill ratings) of user
* **sigma**: sigma value (standard deviation of trueskill ratings) of user 

Columns with NaN value was filled with -1. 

### Feature importance
First image is the feature importance plot without specifying the ```importance_type```. There are some major differences shown in these plots.  
![image](https://user-images.githubusercontent.com/57027695/104830230-286b0700-58c0-11eb-9244-f254664cc009.png)
![image](https://user-images.githubusercontent.com/57027695/104830236-4173b800-58c0-11eb-9e75-1f8b54d065cc.png)

### Cross Validation and train strategy
1. Define an absolute time for the whole database (abs_time = user_id//50 + timestamp//1000)
2. Sort by abs_time. All features mentioned above were engineered such that we will not take data from the future (higher abs_time) into account.
3. Drop first 25% of the data. This data contains some noise, i.e. when nobody studied plenty of questions yet.
4. Take last 25% of data as validation set. 
5. Train the model with remaining 50% of the data. 

### Single Model AUC
We used less than 30 features, but considering that most of the single LGBM models above 0.79 AUC used 40+ features, we did a decent work on focusing on imoportant features.  
LB score: AUC 0.789  
Number of epochs: 6650 (around 15 hours of training in total)

# Transformer
## Encoder
Added below layers with positional encoding. 

### 1) Excercise Related
* **min_delta**: minute difference from between this question and the previous. Cap at 1443 (1 day)
* **day_delta**: day difference from between this question and the previous. Cap at 30
* **month_delta**: month difference from between this question and the previous. Cap at 9
* **tid**: task container id
* **is_with**: if the question is presented with another question. Usually have the same task container
* **c_part**: part, one hot encode and denote skill (listening, reading, part1,2,...)
* **tag1...6**: tags of question (t1 to t6 are the tag of one question, t1 being the most important tag.)
Above embeddings or Dense layer concatenated.

### 2) Content id (cid)
Dense layer

## Decoder
Added below layers with positional encoding.

### 1) Response Related
* **prev_answered_correct**: correctness of previous answer.
* **prior_elapsed**: prior elapsed tiem
* **prior_explained**: prior has explanation
Above embeddings or Dense layer concatenated.

### 2) Answered Correctly

## Concatenate Lecture related Embeddings/Dense
* **num_lect**: number of lecture the user have seen
* **lec_type**: 1 hot encode of most recent lecture type, (llecty1, 2...)
* **lec_h_past**: time since most recent lecture
Above embeddings or Dense layer concatenated.

## Parameters
* WINDOW_SIZE: 100
* EMBED_DIM: 256
* NUM_HEADS: 16

### Cross Validation and train strategy
1. Use first 80% of data as train set and last 20% as validation set. 

### Single Model AUC
AUC 0.786  
SAINT model has plenty of room for improvement, but as 1 epoch took more than 10 hours to train we decided to focus on improving LGBM.

# Inference
### 1. Ensembling two models
Ensembled a single LGBM model and a Transformer model in 0.55 (LGBM) / 0.45 ratio.  
AUC: 0.793

### 2. Ensembling three models
Ensembled two LGBM and a Transformer model. 2nd LGBM was same as the first LGBM but except features related to Trueskill.
When 2 models out of three models predicted that the user is likely to answer correctly, we used the max value among the 3 predictions.
When 2 models out of three models predicted that the user is likely to answer wronly, we used the min value among the 3 predictions.
For remaining cases, we mixed three models in 0.4 (Transformer) / 0.45 (First LGBM) / 0.15 (Second LGBM) ratio.  
AUC: 0.793 (slightly higher than the 1st Inference)

# Training Environment
Our biggest mistake was thinking that all the feature engineering, training, and inferencing process must be done in the Kaggle environment. We were only using Kaggle environment until 2 weeks before the competiton ended, and from then we started to use Google Colab with GPU and 25GB of RAM. Still, there were several times when Colab took GPU from us and didn't give it for several hours as we were constanly using their GPU.

# Possible Improvements
1. Larger window size for the transformer.  
We could check that teams with larger window size for transformer generally got higher score. However, we couldn't test this due to lack of time.
2. Add more key features to LGBM.  
After the competition ended some of the key features that increased AUC a lot were released, and I would like to give them a try. 

