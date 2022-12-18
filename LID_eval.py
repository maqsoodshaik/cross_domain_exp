import torch
model_checkpoint = "/pretrained/wav2vec2-basefrdenlesitptplcommon_voicesame_size_bestmodel"
batch_size = 16
from os import rename
from datasets import load_dataset, load_metric,concatenate_datasets,Dataset,Audio
metric = load_metric("accuracy")
metric_f1 = load_metric("f1")
labels =['fr','de','nl','es','it','pt','pl']
label2id, id2label,label2id_int = dict(), dict(),dict()

for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
    label2id_int[label] = i

from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
max_duration = 10.0  # seconds


dataset_name = "common voice"
configs = ['fr','de','nl','es','it','pt','pl']
list_datasets_validation = []
for i in configs:   
    dataset_validation = load_dataset("common_voice",i,split = "test")
    dataset_validation = dataset_validation.cast_column("audio", Audio(sampling_rate=16000))
    list_datasets_validation.append(dataset_validation)
dataset_validation = concatenate_datasets(
        list_datasets_validation
    )
"""We can then write the function that will preprocess our samples. We just feed them to the `feature_extractor` with the argument `truncation=True`, as well as the maximum sample length. This will ensure that very long inputs like the ones in the `_silence_` class can be safely batched."""
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * max_duration), 
        truncation=True, 
        padding=True
    )
    inputs["labels"] = [label2id_int[image] for image in examples["locale"]]
    return inputs
encoded_dataset_validation = dataset_validation.map(preprocess_function, remove_columns=['locale','client_id', 'path', 'audio', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accent','segment'], batched=True)
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer



import numpy as np

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)
def compute_metrics_f1(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric_f1.compute(predictions=predictions, references=eval_pred.label_ids,average="weighted")


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
best_model = AutoModelForAudioClassification.from_pretrained(
    model_checkpoint
)
model_name_extension = "".join(configs)
model_name = model_checkpoint.split("/")[-1]+model_name_extension+dataset_name
args = TrainingArguments(
    f"{model_name}",#{model_name}arnlpt
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)
trainer = Trainer(
    best_model,
    args,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics
)
trainer_f1 = Trainer(
    best_model,
    args,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics_f1
)
pred= trainer.predict(encoded_dataset_validation)
print(f"Commonvoice_accuracy:{pred}")
print(f"Commonvoice_f1:{trainer_f1.predict(encoded_dataset_validation)}")



