# Riiid AIEd Challenge 2020
## 149th solution (Top 5%, AUC 0.793) of [Riiid! Answer Correctness Prediction](https://www.kaggle.com/c/riiid-test-answer-prediction)

From October 12th, 2020 to January 7th, 2021, AIEd company Riiid! hosted a Kaggle competition that asked participants to create algorithms for "Knowledge Tracing," the modeling of student knowledge over time. The goal was to accurately predict how students will perform on future interactions, based on a [EdNet dataset](https://github.com/riiid/ednet) provided by Riiid.  A total of 3,406 teams/4,412 participants from 100+ countries submitted 64,678 models to explore and tackle AI for education. 

For 2 months, I participated in the competition as a team leader with Harheem Kim, Hung Giang, Sanmaru Um, and Yeseul Gong. We had a lot of limitations: none of us had experience with AI/ML, none of us have particiapated in a Kaggle competition before, and we didn't have any additional hardware support such as GPU machines that a lot of winning teams had. Nevertheless, we were able to grow in a fast pace and secure a Silver medal granted to top 5% of participants with a bit of luck. This repo contains a solution write up for our model, which is an ensemble between a single Light Gradient Boosted Machine model, and a single Encoder-Decoder based Transformer model. Should you have any questions, feel free to contact me at yoonseok@berkeley.edu.
