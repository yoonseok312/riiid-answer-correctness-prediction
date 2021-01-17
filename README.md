# Riiid AIEd Challenge 2020
## 149th solution (Top 5%, AUC 0.793) of [Riiid! Answer Correctness Prediction](https://www.kaggle.com/c/riiid-test-answer-prediction)

From October 12th, 2020 to January 7th, 2021, AIEd company Riiid! hosted a Kaggle competition that asked participants to create algorithms for "Knowledge Tracing," the modeling of student knowledge over time. The goal was to accurately predict how students will perform on future interactions, based on a [EdNet dataset](https://github.com/riiid/ednet) provided by Riiid.  A total of 3,406 teams/4,412 participants from 100+ countries submitted 64,678 models to explore and tackle AI for education. 

For 2 months, I participated in the competition as a team leader with Harheem Kim, Hung Giang, Sanmaru Um, and Yeseul Gong. We had a lot of limitations: none of us had experience with AI/ML, none of us have particiapated in a Kaggle competition before, and we didn't have any additional hardware support such as GPU machines that a lot of winning teams had. Nevertheless, we were able to grow in a fast pace and secure a Silver medal granted to top 5% of participants with a bit of luck. This repo contains a solution write up for our model, which is an ensemble between a single Light Gradient Boosted Machine model, and a single Encoder-Decoder based Transformer model. Should you have any questions, feel free to contact me at yoonseok@berkeley.edu.

## Model explanation
## LightGBM
### Features
* ts_delta: the gap between timestamp of current content with previous content of the same user.
* task_container_id
* prior_question_elapsed_time: in second, rounded
* prior_question_had_explanation
* part
* u_chance: the average correctness of the user until current time
* u_attempts: number of content the user have done
* c_chance: the average correctness of the question until current time
* c_attempts:  number of encounter of that question (all user)
* u_part_chance: the average correctness of the user doing the same part as the question
* u_part_attempts: number of question of the same part the user have done
* u_skill_chance: the average correctness of the user doing the same skill as the question (part < 5: listening, part >= 5: reading)
* u_skill_attempts: number of question of the same skill the user have done
Columns with NaN value was filled with -1. 

### Cross Validation and train strategy
1. Define an absolute time for the whole database (abs_time = user_id//50 + timestamp//1000)
2. Sort by abs_time. All features mentioned above were engineered such that we will not take data from the future (higher abs_time) into account.
3. Drop first 25% of the data. This data contains some noise, i.e. when nobody studied plenty of questions yet.
4. Take last 25% of data as validation set. 
5. Train the model with remaining 50% of the data. 

## Transformer
### Features
* min_delta: minute difference from between this question and the previous. Cap at 1443 (1 day)
* day_delta: day difference from between this question and the previous. Cap at 30
* month_delta: month difference from between this question and the previous. Cap at 9
* cid: content id/question id
* tid: task container id
* prior_elapsed
* prior_explained
* is_with: if the question is presented with another question. Usually have the same task container
* num_lect: number of lecture the user have seen
* lec_type: 1 hot encode of most recent lecture type, (llecty1, 2...)
* lec_h_past: time since most recent lecture
* c_part: part, one hot encode and denote skill (listening, reading, part1,2,...)
* tag1...6: tags of question
* prev_answered_correct: correctness of previous answer.

### Cross Validation and train strategy
1. Use first 80% of data as train set and last 20% as validation set. 

# Inference
## 1. Ensembling two models
## 2. Ensembling three models
TBD
