# %%
from huggingface_hub import login
from dotenv import load_dotenv
import os

# used for runpod instances
os.environ['HF_HOME'] = '/workspace/hfcache/'

load_dotenv()
login(token=os.getenv("HF_HUB_TOKEN"))

# %%
from datasets import load_dataset

# mbti_data = load_dataset("minhaozhang/mbti", split='train')
mbti_data = load_dataset("minhaozhang/mbti")

# %%
# encode the labels for stratified splits
mbti_data = mbti_data.class_encode_column("mbti")
mbti_data = mbti_data.class_encode_column("E-I")
mbti_data = mbti_data.class_encode_column("N-S")
mbti_data = mbti_data.class_encode_column("F-T")
mbti_data = mbti_data.class_encode_column("J-P")

# to train from scratch with entire dataset you don't need any of these splits
# these are only used to test behaviors
# mbti_data = mbti_data['train'].train_test_split(test_size=0.01, stratify_by_column="mbti", seed=0)
# mbti_data = mbti_data["test"]
# mbti_data = mbti_data.train_test_split(test_size=0.01, stratify_by_column="mbti", seed=1)



# %%
# MODEL = "microsoft/Phi-3-mini-4k-instruct" # used for training from scratch
# MODEL = "minhaozhang/Phi-3-mini-4k-instruct-mbti-2" # trained already with 10% data
# TRAINED_MODEL = "Phi-3-mini-4k-instruct-mbti-JP" # newly trained model
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
TRAINED_MODEL = "Llama-3.2-3B-Instruct-MBTI-JP"

# %%
from transformers import AutoTokenizer 

tokenizer = AutoTokenizer.from_pretrained(MODEL)

tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(data):
    return tokenizer(data["body"], truncation=True)

# %%
tokenized_mbti_data = mbti_data.map(preprocess_function, batched=True)
del mbti_data

# remove unnecessary columns
# to train other dimension, change the label to the corresponding column
tokenized_mbti_data = tokenized_mbti_data.remove_columns(['author', 'mbti', 'F-T', "E-I", 'N-S'])
tokenized_mbti_data = tokenized_mbti_data.rename_column('J-P', "label")

# %%
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %%
import evaluate

# use 3 different metrics to evaluate the model
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
balanced_accuracy = evaluate.load("hyperml/balanced_accuracy")
matthews_correlation = evaluate.load("matthews_correlation")

# %%
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy_result = accuracy.compute(predictions=predictions, references=labels)
    f1_result = f1.compute(predictions=predictions, references=labels)
    balanced_accuracy_result = balanced_accuracy.compute(predictions=predictions, references=labels)
    matthews_correlation_result = matthews_correlation.compute(predictions=predictions, references=labels)
    return {**accuracy_result, **f1_result, **balanced_accuracy_result, **matthews_correlation_result}

# %%
# VERY IMPORTANT 
# you have to make sure this is coresponding to the label from the tokenizer and label encoder 
# using 1: J and 0: P will cause NaN in the training which cause the model not to train
id2label = {0: "J", 1: "P"}
label2id = {"J": 0, "P": 1}

# %%
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=2, id2label=id2label, label2id=label2id
)
model.config.pad_token_id = tokenizer.eos_token_id

# %%
import torch
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

steps = 10000 # remember to adjust this

training_args = TrainingArguments(
    output_dir=TRAINED_MODEL,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    eval_strategy="steps",
    save_strategy="steps",
    logging_steps=steps,
    eval_steps=4*steps,
    save_steps=4*steps,
    load_best_model_at_end=True,
    push_to_hub=True,
    optim="adamw_bnb_8bit",
    eval_accumulation_steps=4,
    gradient_accumulation_steps=4, 
    bf16=True,
)

from torch import nn
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        # compute weighted loss
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.6, 0.4]).to('cuda'))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_mbti_data["train"],
    eval_dataset=tokenized_mbti_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()

# %%
trainer.evaluate()
trainer.save_model(TRAINED_MODEL)
trainer.push_to_hub()


