# Model explanation
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
