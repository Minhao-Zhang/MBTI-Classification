# %%
from huggingface_hub import login
from dotenv import load_dotenv
import os
load_dotenv()
login(token=os.getenv("HF_HUB_TOKEN"))

# %%
from datasets import load_dataset

mbti_data = load_dataset("minhaozhang/mbti", split='train')
mbti_data.features

# %%
mbti_data = mbti_data.class_encode_column("mbti")
mbti_data = mbti_data.class_encode_column("E-I")
mbti_data = mbti_data.class_encode_column("N-S")
mbti_data = mbti_data.class_encode_column("F-T")
mbti_data = mbti_data.class_encode_column("J-P")


mbti_data = mbti_data.train_test_split(test_size=0.1, stratify_by_column="mbti", seed=0) # remove the 10% 
mbti_data = mbti_data["train"]
mbti_data = mbti_data.train_test_split(test_size=0.2, stratify_by_column="mbti", seed=0) # 90%  * 20% = 18%
mbti_data = mbti_data["test"]
mbti_data = mbti_data.train_test_split(test_size=0.1, stratify_by_column="mbti", seed=1)


# %%
MODEL = "microsoft/Phi-3-mini-4k-instruct"
TRAINED_MODEL = "Phi-3-mini-4k-instruct-mbti-2"

# %%
from transformers import AutoTokenizer 

tokenizer = AutoTokenizer.from_pretrained(MODEL)

def preprocess_function(data):
    return tokenizer(data["body"], truncation=True)

# %%
tokenized_mbti_data = mbti_data.map(preprocess_function, batched=True)
del mbti_data
# tokenized_mbti_data = tokenized_mbti_data.remove_columns(['body', 'author', 'mbti', 'N-S', 'F-T', 'E-I'])
tokenized_mbti_data = tokenized_mbti_data.remove_columns(['author', 'mbti', 'F-T', "E-I", 'N-S'])
tokenized_mbti_data = tokenized_mbti_data.rename_column('J-P', "label")

# %%
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %%
import evaluate

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

# %%
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy_result = accuracy.compute(predictions=predictions, references=labels)
    f1_result = f1.compute(predictions=predictions, references=labels)
    return {**accuracy_result, **f1_result}

# %%
id2label = {0: "J", 1: "P"}
label2id = {"J": 0, "P": 1}

# %%
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=2, id2label=id2label, label2id=label2id
)

# %%
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

steps = 2000

training_args = TrainingArguments(
    output_dir=TRAINED_MODEL,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    eval_strategy="steps",
    save_strategy="steps",
    logging_steps=steps,
    eval_steps=steps,
    save_steps=steps,
    load_best_model_at_end=True,
    push_to_hub=True,
    optim="adamw_bnb_8bit",
    # optim="adafactor",
    eval_accumulation_steps=4,
    gradient_accumulation_steps=4, 
    # gradient_checkpointing=True,
    # torch_compile=False,
    # bf16=True,
    # fp32=False
    # fp16=False,
    tf32=True,
)

from torch import nn
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
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

# %%
trainer.save_model(TRAINED_MODEL)
trainer.push_to_hub()


