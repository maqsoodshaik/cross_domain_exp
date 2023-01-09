import torch
model_checkpoint = "facebook/wav2vec2-base"
from torch.utils.data import DataLoader
batch_size = 2
num_labels =7
#import sklearn
import sklearn
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices

from os import rename
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer,Wav2Vec2Config,Wav2Vec2ForPreTraining,Wav2Vec2FeatureExtractor
from typing import Dict, List, Optional, Union
from datasets import load_dataset, load_metric,concatenate_datasets,Dataset
metric = load_metric("accuracy")
metric_f1 = load_metric("f1")
labels =["French","German","Dutch","Spanish","Italian","Portuguese","Polish"]
label2id, id2label,label2id_int = dict(), dict(),dict()
from dataclasses import dataclass
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer,Wav2Vec2Config,Wav2Vec2ForPreTraining,Wav2Vec2FeatureExtractor


@dataclass
class DataCollatorForWav2Vec2Pretraining:
    """
    Data collator that will dynamically pad the inputs received and prepare masked indices
    for self-supervised pretraining.

    Args:
        model (:class:`~transformers.Wav2Vec2ForPreTraining`):
            The Wav2Vec2 model used for pretraining. The data collator needs to have access
            to config and ``_get_feat_extract_output_lengths`` function for correct padding.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    model: Wav2Vec2ForPreTraining
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # reformat list to dict and set to pytorch format
        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]

        mask_indices_seq_length = self.model._get_feat_extract_output_lengths(batch["input_values"].shape[-1])
        # make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length)
        # make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )

        features_shape = (batch_size, mask_indices_seq_length)

        # sample randomly masked indices
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.model.config.mask_time_prob,
            self.model.config.mask_time_length,
            attention_mask=batch.get("sub_attention_mask"),
        )

        # sample negative indices
        sampled_negative_indices = _sample_negative_indices(
            features_shape,
            self.model.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )
        batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
        batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)

        return batch
from transformers import AutoFeatureExtractor
feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
ctc_model = Wav2Vec2ForPreTraining.from_pretrained(model_checkpoint)
ctc_model.config.output_hidden_states=True
ctc_model.load_state_dict(torch.load(f"/wop/wav2vec2-basefrenchgermandutchspanishitalianportuguesepolishmultilingual_librispeechsame_size_self_supervised_main_1_balance.pt"))
data_collator = DataCollatorForWav2Vec2Pretraining(
        model=ctc_model, feature_extractor=feature_extractor
    )
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
    label2id_int[label] = i
domains = ["in_domain","outof_domain"]
id2domain = dict()
for i, label in enumerate(domains):
    id2domain[str(i)] = label



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
eval_dataloader_o = DataLoader(encoded_dataset_validation_o, batch_size=batch_size,collate_fn=data_collator)

from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer



def eval_func_train(eval_dataloader, best_model,device,epochs):
    optimizer = torch.optim.SGD(best_model.parameters(), lr=3e-5, momentum=0.9)
    n_correct = 0
    n_samples = 0
    for epoch in range(epochs):
        for batch in eval_dataloader:
            best_model.train()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = best_model(input_values = batch['input_values'],mask_time_indices = batch['mask_time_indices'],sampled_negative_indices = batch['sampled_negative_indices'])
            # out_logits = linear(outputs.hidden_states[-1])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return best_model
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

class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear_1 = torch.nn.Linear(768, num_labels)
    def forward(self, x):
        out = self.linear_1(x)
        return out
linear = LinearRegression()
#load the model
linear.load_state_dict(torch.load(f"/wop/wav2vec2-basefrenchgermandutchspanishitalianportuguesepolishmultilingual_librispeechsame_size_self_supervised_linear_1_balance.pt"))
model_name_extension = "".join(configs_o)

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
    prediced_list = []
    for batch in eval_dataloader:
        best_model.eval()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = best_model(input_values = batch['input_values'])
        out_logits = linear(outputs.hidden_states[-1])
        # loss = criterion(out_logits.mean(dim=1), batch["labels"]) +outputs.loss
        _, predicted = torch.max(out_logits.mean(dim=1), 1)
        prediced_list += predicted.tolist()
        n_samples += 1
        n_correct += (predicted == batch["labels"]).sum().item()
    #calculate F1 score from sklearn
    # breakpoint()
    f1_score = sklearn.metrics.f1_score(encoded_dataset_validation['labels'], prediced_list, average='weighted')
    acc = 100.0 * n_correct / len(encoded_dataset_validation)
    print(f'Accuracy of the network on the validation images: {acc} %')
    print(f'F1-score of the network on the validation images: {f1_score} %')
    #log to wandb
    # wandb.log({f"Accuracy on validation of": acc})
    # wandb.log({f"Loss on validation of": loss})
print(f"multilingual_accuracy")
epochs = 5
ctc_model_org = ctc_model
ctc_model.to(device)
linear.to(device)
ctc_model = eval_func_train(eval_dataloader_o, ctc_model,device,epochs)

# breakpoint()
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
eval_dataloader = DataLoader(encoded_dataset_validation, batch_size=batch_size,collate_fn=data_collator)
ctc_model_org.to(device)
ctc_model_org = eval_func_train(eval_dataloader, ctc_model_org,device,epochs)

# pred= trainer.predict(encoded_dataset_validation)
print(f"voxlingua_accuracy")
with torch.no_grad():
    eval_func(eval_dataloader, ctc_model_org,encoded_dataset_validation)
# print(f"fleaurs_accuracy:{pred}")
# print(f"fleaurs_f1:{trainer_f1.predict(encoded_dataset_validation)}")


