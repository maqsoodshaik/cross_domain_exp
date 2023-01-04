import torch
model_checkpoint = "facebook/wav2vec2-base"
from torch.utils.data import DataLoader
batch_size = 16
num_labels =7
from os import rename
from datasets import load_dataset, load_metric,concatenate_datasets,Dataset
metric = load_metric("accuracy")
metric_f1 = load_metric("f1")
labels =["French","German","Dutch","Spanish","Italian","Portuguese","Polish"]
label2id, id2label,label2id_int = dict(), dict(),dict()

for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
    label2id_int[label] = i
domains = ["in_domain","outof_domain"]
id2domain = dict()
for i, label in enumerate(domains):
    id2domain[str(i)] = label
from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
max_duration = 10.0  # seconds


dataset_name_o = "multilingual_librispeech"
configs_o = ['french', 'german', 'dutch','spanish','italian','portuguese','polish']
list_datasets_validation_o = []
for val,i in enumerate(configs_o):   
    dataset_validation = load_dataset("facebook/multilingual_librispeech",i,split = "test")
    dataset_validation = dataset_validation.add_column("labels",[val]*len(dataset_validation))
    list_datasets_validation_o.append(dataset_validation)
dataset_validation_o = concatenate_datasets(
        list_datasets_validation_o
    )
"""We can then write the function that will preprocess our samples. We just feed them to the `feature_extractor` with the argument `truncation=True`, as well as the maximum sample length. This will ensure that very long inputs like the ones in the `_silence_` class can be safely batched."""

def preprocess_function_o(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * max_duration), 
        truncation=True,
        padding=True 
    )
    return inputs
encoded_dataset_validation_o = dataset_validation_o.map(preprocess_function_o, remove_columns=['file','audio','text','speaker_id','chapter_id','id'], batched=True)
encoded_dataset_validation_o.set_format("torch")
eval_dataloader_o = DataLoader(encoded_dataset_validation_o, batch_size=batch_size)

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
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer,Wav2Vec2Config,Wav2Vec2ForPreTraining,Wav2Vec2FeatureExtractor

ctc_model = Wav2Vec2ForPreTraining.from_pretrained(model_checkpoint)
ctc_model.config.output_hidden_states=True
ctc_model.load_state_dict(torch.load(f"/wop/wav2vec2-basefrdenlesplptitvoxlinguasame_size_self_supervised_main.pt"))
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear_1 = torch.nn.Linear(768, 256)
        self.linear_2 = torch.nn.Linear(256, num_labels)
    def forward(self, x):
        out = self.linear_1(x)
        out = self.linear_2(out)
        return out
linear = LinearRegression()
#load the model
linear.load_state_dict(torch.load(f"/wop/wav2vec2-basefrdenlesplptitvoxlinguasame_size_self_supervised_linear.pt"))
model_name_extension = "".join(configs_o)
ctc_model.to(device)
linear.to(device)
model_name = model_checkpoint.split("/")[-1]+model_name_extension+dataset_name_o
# args = TrainingArguments(
#     f"{model_name}",#{model_name}arnlpt
#     evaluation_strategy = "epoch",
#     save_strategy = "epoch",
#     learning_rate=3e-5,
#     per_device_train_batch_size=batch_size,
#     gradient_accumulation_steps=4,
#     per_device_eval_batch_size=batch_size,
#     num_train_epochs=5,
#     warmup_ratio=0.1,
#     logging_steps=10,
#     load_best_model_at_end=True,
#     metric_for_best_model="accuracy",
# )
# trainer = Trainer(
#     best_model,
#     args,
#     tokenizer=feature_extractor,
#     compute_metrics=compute_metrics
# )
# trainer_f1 = Trainer(
#     best_model,
#     args,
#     tokenizer=feature_extractor,
#     compute_metrics=compute_metrics_f1
# )
# for batch in trainer.get_train_dataloader():
#     break
# from scipy.io.wavfile import write
# write('output_sounddevice.wav', 16000, np.array(batch["input_values"][0]))
# print(f"after loading model:{trainer.evaluate()}")
# pred_o= trainer.predict(encoded_dataset_validation_o)
@torch.no_grad()
def eval_func(eval_dataloader, best_model,encoded_dataset_validation):
    n_correct = 0
    n_samples = 0
    for batch in eval_dataloader:
        best_model.eval()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = best_model(input_values = batch['input_values'])
        out_logits = linear(outputs.hidden_states[-1])
        # loss = criterion(out_logits.mean(dim=1), batch["labels"]) +outputs.loss
        _, predicted = torch.max(out_logits.mean(dim=1), 1)
        n_samples += 1
        n_correct += (predicted == batch["labels"]).sum().item()
    acc = 100.0 * n_correct / len(encoded_dataset_validation)
    print(f'Accuracy of the network on the validation images: {acc} %')
    #log to wandb
    # wandb.log({f"Accuracy on validation of": acc})
    # wandb.log({f"Loss on validation of": loss})
print(f"multilingual_accuracy")
with torch.no_grad():
    eval_func(eval_dataloader_o, ctc_model,encoded_dataset_validation_o)
# print(f"multilingual_accuracy:{pred_o}")
# print(f"Multilingual_f1:{trainer_f1.predict(encoded_dataset_validation_o)}")





import datasets

from pathlib import Path
datasets.config.DOWNLOADED_DATASETS_PATH = Path('/corpora/voxlingua/')


###fleaurs
# configs = ['fr','de','nl']
labels_f =["French","German","Dutch","Spanish","Italian","Portuguese","Polish"]
label2id_f, id2label_f,label2id_int_f = dict(), dict(),dict()

for i, label in enumerate(labels_f):
    label2id_f[label] = str(i)
    id2label_f[str(i)] = label
    label2id_int_f[label] = i
dataset_name = "voxlingua"
configs = ['fr','de','nl','es','it','pt','pl']
# configs = ['ru_ru','pl_pl','uk_ua']
list_datasets_validation = []
for index,i in enumerate(configs):   
    dataset_validation = load_dataset("/corpora/voxlingua/",data_dir=i,split = "test")
    # dataset_validation = Dataset.from_dict(dataset_validation[:20])
    dataset_validation = dataset_validation.add_column("labels",[index]*len(dataset_validation))
    list_datasets_validation.append(dataset_validation)
dataset_validation = concatenate_datasets(
        list_datasets_validation
    )

def preprocess_function_f(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * max_duration), 
        truncation=True,
        padding=True 
    )
    # inputs["labels"] = [label2id_int_f[image] for image in examples["label"]]
    # inputs["labels"] = examples["label"]
    return inputs
encoded_dataset_validation = dataset_validation.map(preprocess_function_f, remove_columns=["audio","label"], batched=True)
encoded_dataset_validation.set_format("torch")
eval_dataloader = DataLoader(encoded_dataset_validation, batch_size=batch_size)

# pred= trainer.predict(encoded_dataset_validation)
print(f"fleaurs_accuracy")
with torch.no_grad():
    eval_func(eval_dataloader, ctc_model,encoded_dataset_validation)
# print(f"fleaurs_accuracy:{pred}")
# print(f"fleaurs_f1:{trainer_f1.predict(encoded_dataset_validation)}")


