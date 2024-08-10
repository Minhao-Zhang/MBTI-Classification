# MBTI Classifier - A Feasibility Study

Classifying someone's MBTI type based on their text data.

此文章有[中文版本](./README_ZH.md)。

## Data Preparation

### Data Curation 

For this project, I created a custom dataset that is publicly available. 
The raw data, originally uploaded to [Zenodo](https://zenodo.org/records/1482951) by Dylan Storey, was sourced from Reddit using Google Big Query. 
The data consists of posts from users who have self-identified their MBTI type. 
You can download the raw data by running the following command:

```bash
./setup.sh
```

This command will also create the necessary temporary directories for data processing.

The data underwent extensive cleaning, including:
- Lowercasing all text.
- Removing all URLs.
- Excluding posts containing non-English characters.
- Stripping Reddit-specific links such as `r/abcd` and `u/abcd`.
- Removing all special characters except for `?` and `!`.
- Filtering out posts with fewer than 20 characters or more than 3,000 characters.

For a detailed overview of the cleaning steps, refer to [clean_data.py](./preprocessing/clean_data.py).

During the cleaning process, I identified some discrepancies in the data:
- Some authors were associated with multiple MBTI types.
- Some posts were duplicated by the same author.

To address these issues, I removed all duplicates and excluded posts from a user who had posted inappropriate content, leading to their ban by several moderators. 
All posts by this user were removed to maintain data integrity. 
These steps are implemented in [remove_duplicate.py](./preprocessing/remove_duplicate.py).

### Data Summary

The final dataset comprises 13 million rows and 3 columns, representing 11,773 unique authors. 
Each row contains a single post, along with the author's name and MBTI type. 
Additionally, a separate dataset with unique authors and their MBTI types is provided.

This dataset serves as a foundational resource for MBTI classification. 
It is available for download on [Kaggle](https://www.kaggle.com/datasets/minhaozhang1/reddit-mbti-dataset).

### Pre-processing 

Given the nature of chat messages, the length of each message varies significantly. 
Very short messages may not provide sufficient information for accurate personality prediction. 
To address this, I combined shorter messages to ensure a minimum length of 700 characters (including spaces) and a maximum of 1,000 characters. 
Detailed steps are provided in [combine_short_text.py](./preprocessing/combine_short_text.py).

Although counting characters is effective for creating a balanced dataset, it may introduce challenges when using LLMs' tokenization, which is based on words rather than characters. 
To mitigate this, I further processed the data to achieve a narrower range of word lengths. 
The steps are detailed in [evenout_word_length.py](./preprocessing/evenout_word_length.py), and the dataset can be accessed on [Hugging Face](https://huggingface.co/datasets/minhaozhang/mbti).

### Baseline Score

With the final dataset prepared, we can explore the distribution of MBTI types within it. 
A majority classifier can be used to establish baseline performance. 
More information can be found in [eda.ipynb](./preprocessing/eda.ipynb).

| Type | Accuracy | F1 Score |
| ---- | -------- | -------- |
| E-I  | 0.78858  | 0.88179  |
| N-S  | 0.92603  | 0.96160  |
| F-T  | 0.53863  | 0.70014  |
| J-P  | 0.59189  | 0.74363  |


## Machine Learning Approach 

### Previous Work

There have been various efforts to predict personality traits based on text data. 
One of the most well-known examples is the questionnaire provided by [16personalities.com](https://www.16personalities.com/), which classifies an individual's MBTI type based on their responses. 
However, predicting personality from natural conversation data presents a much greater challenge. 
A notable attempt in this area is documented by Ryan et al. (2023), where they used a dataset from Kaggle to predict MBTI types from text data.

Ryan et al. employed a traditional machine learning approach, utilizing a TF-IDF vectorizer combined with classifiers like CatBoost, along with the SMOTE technique to balance the data. 
Despite demonstrating some improvement with SMOTE, their model's performance was underwhelming. 
For instance, their best F1 score for the binary classification of I/E was 0.8389. 
However, given the I/E distribution of 6676/1999, a simple majority classifier could achieve an F1 score of 0.86978. 
This indicates that their model did not outperform a basic majority classifier.

### My Approach

I replicated the approach of Ryan et al. (2023) using my own cleaned and significantly larger dataset. 
While I adopted a similar data preprocessing strategy, I experimented with several gradient boosting classifiers, including CatBoost, XGBoost, and LightGBM. 
Although my data distribution differed from theirs, the results were still disappointing, with my best F1 score equal to the majority classifier. 
The detailed training and evaluation process is documented in [train_model.ipynb](./ml/train_model.ipynb).

These unsatisfactory results might be attributed to the inherent complexity of MBTI classification or the limitations of traditional machine learning techniques. 
Therefore, I will explore the potential of large language models to see if they can enhance performance.


<table>
  <tr>
    <th>Type</th>
    <th>Metric</th>
    <th>Baseline</th>
    <th>XGBoost</th>
    <th>CatBoost</th>
    <th>LightGBM</th>
  </tr>
  <tr>
    <td rowspan="2">E-I</td>
    <td>Accuracy</td>
    <td>0.7886</td>
    <td>0.7891</td>
    <td>0.7889</td>
    <td>0.7890</td>
  </tr>
  <tr>
    <td>F1 Score</td>
    <td>0.8818</td>
    <td>0.8819</td>
    <td>0.8820</td>
    <td>0.8820</td>
  </tr>
  <tr>
    <td rowspan="2">N-S</td>
    <td>Accuracy</td>
    <td>0.9260</td>
    <td>0.9263</td>
    <td>0.9261</td>
    <td>0.9262</td>
  </tr>
  <tr>
    <td>F1 Score</td>
    <td>0.9616</td>
    <td>0.9617</td>
    <td>0.9616</td>
    <td>0.9617</td>
  </tr>
  <tr>
    <td rowspan="2">F-T</td>
    <td>Accuracy</td>
    <td>0.5386</td>
    <td>0.6284</td>
    <td>0.6248</td>
    <td>0.6234</td>
  </tr>
  <tr>
    <td>F1 Score</td>
    <td>0.7001</td>
    <td>0.6794</td>
    <td>0.6794</td>
    <td>0.6781</td>
  </tr>
  <tr>
    <td rowspan="2">J-P</td>
    <td>Accuracy</td>
    <td>0.5919</td>
    <td>0.6162</td>
    <td>0.6116</td>
    <td>0.6116</td>
  </tr>
  <tr>
    <td>F1 Score</td>
    <td>0.7436</td>
    <td>0.3248</td>
    <td>0.2596</td>
    <td>0.2619</td>
  </tr>
</table>


## Large Language Models Approach

### Getting Started

I decided to use Microsoft's recently released [Phi-3](https://azure.microsoft.com/en-us/blog/introducing-phi-3-redefining-whats-possible-with-slms/) model for this project. 
Although Phi-3 recently became available for fine-tuning on Azure AI Studio, I opted to use the Hugging Face [Transformers](https://huggingface.co/transformers/) library for its greater flexibility in fine-tuning.

To begin, I followed the [sequence classification tutorial](https://huggingface.co/docs/transformers/en/tasks/sequence_classification) on Hugging Face, which demonstrates fine-tuning a model using Google's BERT.
 While the tutorial provides a clear structure for fine-tuning, I replaced BERT with the more advanced Phi-3 model. 
 However, this introduced higher computational requirements. 
 Even with the smallest [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) model, fine-tuning on my PC was impractical. 
 To overcome this, I utilized a cloud service with an A100 GPU for the fine-tuning process.

### Reducing GPU Memory Usage

Despite the A100 GPU's superior capabilities, I still needed to employ strategies to reduce vRAM usage. 
Using Hugging Face's [Model Memory Estimator](https://huggingface.co/spaces/hf-accelerate/model-memory-usage), I found that the model, combined with the `Adam` optimizer and `float32` dtype, required 57GB of peak vRAM. 
This, along with the space needed for the data, would exceed the 80GB vRAM capacity of the A100 GPU. 
Therefore, I implemented several strategies to optimize memory usage.

All of these strategies were adapted from the [Transformers](https://huggingface.co/docs/transformers/perf_train_gpu_one) tutorials.

| Method/Tool                            | Methods I Employed                                |
| -------------------------------------- | ------------------------------------------------- |
| Batch size choice                      | Yes, to reduce vRAM usage                         |
| Gradient accumulation                  | Yes, to effectively increase the batch size       |
| Gradient checkpointing                 | No, it decreases training speed by 20%            |
| Mixed precision training               | Yes, used `tf32` to increase training speed       |
| torch_empty_cache_steps                | No, it decreases training speed by 10%            |
| Optimizer choice                       | Yes, used `adamw_bnb_8bit` to reduce memory usage |
| Data preloading                        | Yes, by default                                   |
| DeepSpeed Zero                         | No, I couldn't set up the environment             |
| torch.compile                          | No, I couldn't set up the environment             |
| Parameter-Efficient Fine Tuning (PEFT) | No, I couldn't set up the environment             |

One interesting observation was that when I attempted to use the `bf16` dtype to further reduce memory usage, the model's loss output became `nan`. 
I found a similar issue discussed in a [Hugging Face forum post](https://discuss.huggingface.co/t/training-loss-0-0-validation-loss-nan/27950), although it involved a different model. 
According to the [config.json](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/config.json#L31) file, the model was trained using the `bf16` dtype, but this did not resolve the issue. 
I have yet to find a solution and welcome any insights.

### Fine-Tuning the LLM 

Building on the sentiment analysis tutorial, I adapted the code to fit the MBTI classification task. 
Since the MBTI can be broken down into four binary classification tasks, I treated it as such rather than a single 16-class classification task. 
I started with the J-P type classification, as its distribution is relatively balanced, though slightly skewed. 
Using a 1% subset of the training split, I began fine-tuning the model. 
Initially, it appeared that the model was learning effectively.

![screenshot](./figs/run_screenshot_0.png)

The model's accuracy surpassed that of the majority classifier. 
However, this could have been due to random chance. 
After optimizing my code (I realized I was not using `tf32` as intended) and increasing the dataset size to 10% of the training split, the results were less promising.

![screenshot](./figs/run_screenshot_1.png)

While the fine-tuned model showed some ability to classify the J-P type, it failed to outperform the majority classifier, indicating that it had learned to predict the majority class exclusively. 
Recognizing this issue, I paused the training to diagnose the problem.

### Adding a System Prompt

At this stage, I realized that identifying patterns in MBTI types might be more challenging for the LLM compared to simpler tasks like sentiment analysis. 
The complexity of MBTI classification requires more nuanced understanding. 
To aid the model, I introduced a semi-system prompt before each input text:

```text
You are an MBTI expert trying to identify, using the text below, which personality type they are. 
You are only predicting Judging-Perceiving (J-P) for this task. Predict 0 for J and 1 for P.
Learn the personality, not the proportion of data. Here is the text:
```

The intent was to guide the model on what to focus on. 
Unfortunately, this adjustment did not result in any performance improvement.

### Other MBTI Dimensions

Given the challenges with the J-P classification, I decided to experiment with other MBTI dimensions. 
I chose the Feeling-Thinking (F-T) dimension next, as it is the most balanced. 
Initially, I did not include the system prompt in the input.

I used a 5% subset of the training split to start fine-tuning the model.

![Screenshot](./figs/run_screenshot_2.png)

The model achieved a higher accuracy score, but the F1 score was lower. 
A noticeable drop in both accuracy and F1 score occurred around the 8000-step mark, which coincided with the start of a new epoch, indicating that the model was re-training on the same data. 
By 9000 steps, the model's training loss was extremely low, while the validation loss remained high, signaling clear overfitting. 
This suggests that limiting the training to a single epoch may be necessary, as the model overfits as soon as the data is reintroduced.

I attempted to train the model on other MBTI dimensions, but the results were similar.

### Re-processing the Data

Initially, I removed all punctuation except for `?` and `!`, and converted all text to lowercase to facilitate word2vec model training. 
Additionally, the posts in my dataset were limited to around 200 words each, which might not have provided enough context for the LLM to detect speech patterns. 
I increased the word count per post to 400, although this reduced the overall number of samples. 
The data was reprocessed as detailed in [re_process_data.ipynb](./preprocessing/re_process_data.ipynb), and the model was fine-tuned again. 
Despite these adjustments, the model still did not perform better than a majority classifier.


## Treating Imbalanced Data

### Overview

Imbalanced data is a common challenge in classification tasks, and even powerful models like LLMs are not immune to this issue. 
When faced with imbalanced data, models often default to predicting the majority class, leading to poor performance on the minority class. 
To address this, several methods can be employed:

- Over-sampling the minority class / Under-sampling the majority class
- SMOTE (Synthetic Minority Over-sampling Technique)
- Class weight balancing

### Under-sampling the Majority Class

As observed in the previous sections, the model exhibited overfitting when reintroduced to the training data. 
Given this, over-sampling the minority class could exacerbate overfitting, so I opted for under-sampling the majority class instead. 
With ample data at my disposal, I could afford to reduce the size of the majority class without compromising overall dataset size. 
A straightforward approach to under-sampling involved randomly selecting a subset of the majority class to match the size of the minority class.

I began with the J-P dimension, which is somewhat imbalanced but not excessively so. 
After splitting out an additional validation set from the training subset, I under-sampled the majority class. 
However, this strategy did not lead to any improvement in model performance. 
Although the training loss changed, indicating that the data distribution was different, the accuracy and F1 scores remained comparable to those of the majority classifier. 
This outcome suggests that my basic implementation of under-sampling may not have been sufficient or that MBTI classification inherently poses challenges for this method.

### Synthetic Minority Over-sampling Technique (SMOTE)

SMOTE generates synthetic samples for the minority class, and it's accessible through the `imbalanced-learn` library. 
A discussion on [Hugging Face's forum](https://discuss.huggingface.co/t/how-to-apply-smote-to-a-dataset/27876) explored using SMOTE with a custom dataloader for text data. 
However, based on my research, I concluded that SMOTE might not be suitable for generating synthetic text data. 
Although Ryan et al. (2023) employed SMOTE in their work and observed performance gains, their model still did not surpass the majority classifier. 
This led me to consider alternative approaches for handling imbalanced data.

### Class Weight Balancing

Another technique I explored was class weight balancing, as suggested in a [forum post](https://discuss.huggingface.co/t/how-can-i-use-class-weights-when-training/1067). 
This method involves scaling the loss associated with the minority class by the ratio of the majority class to the minority class. 
For the J-P dimension, which has a 60-40 split, I inversely set the weights based on the class proportions.

Unfortunately, this approach did not significantly enhance the model's performance. 
While the training loss initially indicated that the model was attempting to learn patterns within the text, it soon defaulted to predicting the majority class consistently.

However, there was a promising development:

![Screenshot](./figs/run_screenshot_3.png)

When I increased the training size, the model began to show signs of learning. 
The accuracy surpassed that of the majority classifier, which is an encouraging indicator. 
I plan to continue training this model to see if further improvements can be achieved.


## Thoughts and Future Work

### Thoughts

MBTI classification based on text data presents unique challenges that set it apart from more straightforward tasks like sentiment analysis. Unlike sentiment, which often has clear indicators within the text (e.g., specific keywords or phrases), MBTI classification requires deeper analysis of context, speech habits, and the speaker's intentions. This subtlety makes it difficult even for advanced models to accurately predict personality types. The task demands a nuanced understanding of language and behavior, which LLMs might still struggle to fully grasp.

### Future Work

There are several areas where this project could be further improved:

1. **Use a Better and Longer Dataset**:
   - In this project, each post was limited to around 200-400 words to maintain a large enough dataset. However, longer texts could potentially provide more context and thus be more informative for the model. Combining shorter texts into longer ones could decrease the number of samples, but it might also introduce biases if many long posts are authored by the same individuals. A better dataset with longer posts and more diverse samples would likely enhance the model's performance. 

1. **Use a Better LLM Model**:
   - While Phi-3 is a relatively new and promising model, it's still comparatively small, with 3.8 billion parameters and a 4k token context. Larger state-of-the-art models, like Llama3.1, which can have up to 405 billion parameters and 128k token context, might be better suited for this task. These larger models could extract more nuanced information from the text and potentially deliver better performance. Unfortunately, due to computational constraints, fine-tuning such large models was not feasible in this project.

2. **Use a Better Fine-Tuning Strategy**:
   - The current approach utilized the `AutoModelForSequenceClassification` from the Hugging Face Transformers library, which is a good starting point. However, a custom training loop using PyTorch might offer more flexibility and allow for fine-tuning that is better tailored to the specific challenges of MBTI classification.

3. **Use a Better Data Pre-processing Strategy**:
   - The data pre-processing strategy employed in this project was straightforward but might not be optimal for this particular task. While common NLP techniques were applied, more sophisticated pre-processing strategies could be developed, potentially incorporating insights from psychology and linguistics. This could involve more nuanced handling of language structures and contextual elements, which might better support the model in recognizing the subtleties of personality types. 

## References

Ryan, Gregorius, Pricillia Katarina, and Derwin Suhartono. 2023. "MBTI Personality Prediction Using Machine Learning and SMOTE for Balancing Data Based on Statement Sentences" Information 14, no. 4: 217. https://doi.org/10.3390/info14040217