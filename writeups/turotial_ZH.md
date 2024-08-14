# 如何微调大语言模型（LLMs）用文本来预测MBTI

## 介绍

文本分类是自然语言处理（NLP）中的常见任务。它是根据文本内容将标签分配给一段文本的过程。例如，将电影评论分类为正面或负面，或者将新闻文章分类为体育、政治或娱乐。在大型语言模型（LLMs）出现之前，文本分类是通过传统的机器学习方法进行的，并且需要大量的特征工程。使用像Word2Vec、GloVe和FastText这样的词嵌入将文本转换为数值向量。然后，将这些向量输入到像逻辑回归、支持向量机或随机森林这样的机器学习模型中进行分类。这个过程可能很慢，并且需要大量的手动工作才能获得良好的结果。

借助LLMs的强大功能，某些文本分类任务可以通过几行代码解决。在本教程中，我们将使用Hugging Face Transformers库来训练一个文本分类模型，以根据文本数据分类一个人的**MBTI**。

## 什么是MBTI？

迈尔斯-布里格斯类型指标（MBTI）是一种性格测试，将人们分为16种不同的性格类型。每种性格类型是四个字母的组合，每个字母代表一个不同的性格方面。

这四个方面是：
1. **外向（E）**与**内向（I）**
2. **感觉（S）**与**直觉（N）**
3. **思考（T）**与**情感（F）**
4. **判断（J）**与**知觉（P）**

## 数据集

我将使用一个我自己整理的MBTI数据集。有关更多信息，请点击[此处](https://github.com/Minhao-Zhang/MBTI-Classification)。

简而言之，这是一个包含300万行的数据库，包含以下列：
- author: 文本的作者
- body: 文本数据
- mbti: 作者的MBTI类型
- E-I: MBTI中的外向（E）与内向（I）方面
- N-S: MBTI中的直觉（N）与感觉（S）方面
- F-T: MBTI中的情感（F）与思考（T）方面
- J-P: MBTI中的判断（J）与知觉（P）方面

## 设置环境

尽管现在可以在CPU上运行许多LLM，但对它们进行微调仍然没有GPU不可行。因此，本教程需要一块具有超过40GB显存的GPU。我建议使用能够提供已安装CUDA和cuDNN的Ubuntu虚拟机的云计算服务。否则，你应该首先确保你的机器上安装了适当版本的CUDA和cuDNN。有关安装过程的更多信息，请参阅[这里](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)。

一切准备好后，我们可以开始创建一个新的Python环境并安装必要的库。

```bash
conda create -n mbti-tuning python=3.12.4
conda activate mbti-tuning
```

接着，我们需要安装以下库：

```bash
pip install numpy scikit-learn python-dotenv datasets transformers evaluate accelerate pytorch
```

## 加载数据集

为了保存模型，我们需要在Hugging Face网站上创建一个账户并获取访问密钥。我们可以将访问密钥保存在项目根目录下的`.env`文件中。

```python
from huggingface_hub import login
from dotenv import load_dotenv
import os
load_dotenv()
login(token=os.getenv("HF_HUB_TOKEN"))
```

然后，你可以使用`datasets`库加载数据集。以下代码将从Hugging Face下载数据集或从缓存中加载数据集。

```python
from datasets import load_dataset

mbti_data = load_dataset("minhaozhang/mbti", split='train')
```

为了确保有良好的训练-验证划分，我们将使用分层抽样。因此，我们需要先对列进行编码。

```python
mbti_data = mbti_data.class_encode_column("mbti")
mbti_data = mbti_data.class_encode_column("E-I")
mbti_data = mbti_data.class_encode_column("N-S")
mbti_data = mbti_data.class_encode_column("F-T")
mbti_data = mbti_data.class_encode_column("J-P")
```

由于训练集包含大约230万行数据，训练所有数据可能会很慢。因此，我们将只使用10%的数据进行训练，再使用其20%的数据进行验证。

```python
mbti_data = mbti_data.train_test_split(test_size=0.1, stratify_by_column="mbti", seed=0)
mbti_data = mbti_data["test"] # 丢弃90%的数据
mbti_data = mbti_data.train_test_split(test_size=0.2, stratify_by_column="mbti", seed=1)
```

## 定义模型

与Hugging Face上使用较旧的BERT模型的教程不同，我将使用微软于2024年发布的Phi-3模型。

```python
MODEL = "microsoft/Phi-3-mini-4k-instruct"
TRAINED_MODEL = "Phi-3-mini-4k-instruct-mbti"
```

然后，我们可以定义模型和分词器。

```python
from transformers import AutoTokenizer 

tokenizer = AutoTokenizer.from_pretrained(MODEL)

def preprocess_function(data):
    return tokenizer(data["body"], truncation=True)

tokenized_mbti_data = mbti_data.map(preprocess_function, batched=True)
del mbti_data # 节省内存
```

为了简化问题，我将只对MBTI的J-P方面进行分类。在这个数据集中，J-P方面的比例是60-40，它有一定的不平衡但并没有严重偏斜。因此，我将移除所有其他列。

```python
tokenized_mbti_data = tokenized_mbti_data.remove_columns(['author', 'mbti', 'F-T', "E-I", 'N-S'])
tokenized_mbti_data = tokenized_mbti_data.rename_column('J-P', "label")
```

为了简化数据批处理并通过动态填充令牌提高效率，我们将使用`DataCollator`。

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

为了评估我们的模型，`evaluate`包提供了许多开箱即用的指标。

```python
import evaluate

# 使用3种不同的指标来评估模型
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
matthews_correlation = evaluate.load("matthews_correlation")

import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy_result = accuracy.compute(predictions=predictions, references=labels)
    f1_result = f1.compute(predictions=predictions, references=labels)
    matthews_correlation_result = matthews_correlation.compute(predictions=predictions, references=labels)
    return {**accuracy_result, **f1_result, **matthews_correlation_result}
```

我们还需要一个标签和ID之间的映射，以便模型在训练中进行损失计算。不正确的映射会导致损失计算中的NaN，使模型无法学习。

```python
id2label = {0: "J", 1: "P"}
label2id = {"J": 0, "P": 1}
```

现在，我们可以使用`transformers`中的`AutoModel`来定义模型。

```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=2, id2label=id2label, label2id=label2id
)
```
## 微调LLMs

为了使用提供的`AutoModelForSequenceClassification`类，我们需要定义训练参数。

```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

training_args = TrainingArguments(
    output_dir=TRAINED_MODEL,
    learning_rate=2e-5,
    per_device_train_batch_size=4, # 根据显存调整
    per_device_eval_batch_size=4, # 根据显存调整
    num_train_epochs=1, 
    weight_decay=0.01,
    eval_strategy="steps", # 根据数据大小调整
    save_strategy="steps", # 根据数据大小调整
    logging_steps=1000, # 根据数据大小调整
    eval_steps=1000, # 根据数据大小调整
    save_steps=1000, # 根据数据大小调整
    load_best_model_at_end=True,
    push_to_hub=True,
    optim="adamw_bnb_8bit",
    eval_accumulation_steps=2, # 根据显存调整
    gradient_accumulation_steps=2, # 根据显存调整
    tf32=True,
)
```

在这些训练参数中，使用了几种方法来加速训练过程并减少内存的使用。
- 批量大小设置为4，以减少内存使用。
- `optim="adamw_bnb_8bit"`：使用AdamW和ByteNetBlock 8位量化，这将减少显存使用量4倍。有关更多信息，请参见[这里](https://huggingface.co/docs/transformers/perf_train_gpu_one#optimizer-choice)。
- `gradient_accumulation_steps=2`和`eval_accumulation_steps=2`有效地将批量大小增加到8。
- `tf32=True`，使用TensorFloat32进行训练。

![precision-comparison](float-precision-comparison.png)

从这个图像中可以看到，使用tf32将牺牲精度以换取更小的内存使用率。TF32使用19位而不是32位来表示每个浮点数，从而减少了40%的内存使用。根据[NVIDIA](https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/)的研究，使用tf32训练的模型与使用fp32训练的模型性能非常相似。

> 你可以注意到我们没有使用bf16来微调模型。根据模型配置，Phi3模型实际上是使用bf16进行训练的。然而，由于某些未知的原因，如果在微调过程中使用bf16，损失将变为NaN。

现在，我们可以定义训练器并开始训练模型。

```python
from torch import nn
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # 前向传递
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # 计算自定义损失
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
```

在这里，我们编写了一个具有特殊损失函数的自定义训练器。如前所述，我们的数据不平衡，大致为60-40的比例。因此，我们使用了权重平衡方法，以防止模型变成一个主要分类器。

最后，我们终于可以开始训练模型了。

```python
trainer.train()
trainer.evaluate()
trainer.save_model(TRAINED_MODEL)
trainer.push_to_hub()
```

这些代码将训练模型并将训练好的模型推送到你的Hugging Face仓库。

## 结果

如果将上述内容放入Jupyter Notebook中，你将看到类似于下图的结果。

![run screenshot](run_screenshot.png)

随着训练损失和验证损失稳步下降，准确率在增长。在这个例子中，我们的模型在验证集上达到了约0.65的准确率。这个结果并不是特别好，但考虑到我们只使用了10%的数据和一个小型模型，这是一个不错的结果。如果你想要更好的结果，你可以尝试使用更大的模型、更多的数据和更多的训练时间。

如果你想要查看完整的代码，请点击[这里](https://github.com/Minhao-Zhang/MBTI-Classification/)。