from turtle import color
import torch
model_checkpoint = "facebook/wav2vec2-base"
batch_size = 32
from os import rename
from datasets import load_dataset, load_metric,concatenate_datasets,Dataset
dataset_name = "fleurs"
configs = ['fr_fr','de_de','nl_nl']
list_datasets_validation = []
for i in configs:   
    dataset_validation = load_dataset("google/fleurs",i,split = "validation")
    dataset_validation = Dataset.from_dict(dataset_validation[:20])
    list_datasets_validation.append(dataset_validation)
dataset_validation = concatenate_datasets(
        list_datasets_validation
    )
metric = load_metric("accuracy")
labels =["French","German","Dutch"]
label2id, id2label,label2id_int = dict(), dict(),dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
    label2id_int[label] = i
from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
max_duration = 3.0  # seconds

"""We can then write the function that will preprocess our samples. We just feed them to the `feature_extractor` with the argument `truncation=True`, as well as the maximum sample length. This will ensure that very long inputs like the ones in the `_silence_` class can be safely batched."""

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * max_duration), 
        truncation=True, 
    )
    inputs["labels"] = [label2id_int[image] for image in examples["language"]]
    return inputs
encoded_dataset_validation = dataset_validation.map(preprocess_function, remove_columns=["id","num_samples", "path", "audio", "transcription", "raw_transcription", "gender", "lang_id", "language", "lang_group_id"], batched=True)
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

model_name_extension = "".join(configs)
model_name = model_checkpoint.split("/")[-1]+model_name_extension+dataset_name

args = TrainingArguments(
    f"{model_name}",#{model_name}arnlpt
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=20,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)
import numpy as np

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


best_model = AutoModelForAudioClassification.from_pretrained(
    f"wav2vec2-basefr_frde_denl_nlfleurs_fleurs_bestmodel"
)

trainer = Trainer(
    best_model,
    args,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics
)
# for batch in trainer.get_train_dataloader():
#     break
# from scipy.io.wavfile import write
# write('output_sounddevice.wav', 16000, np.array(batch["input_values"][0]))
# print(f"after loading model:{trainer.evaluate()}")
pred= trainer.predict(encoded_dataset_validation)
print(pred)
best_model = best_model.wav2vec2
inp = encoded_dataset_validation["input_values"][::2]
labels_p = encoded_dataset_validation["labels"][::2]
pred = best_model(torch.tensor(inp))
pred = pred.last_hidden_state.reshape(pred.last_hidden_state.shape[0],-1)
pred = pred.detach().numpy()
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pred =pca.fit_transform(pred)

import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots()
 
# Plot scaled features
xdata = pred[:,0]
ydata = pred[:,1]
 
# Plot 3D plot
scatter =ax.scatter(xdata, ydata,c=labels_p)
 
# Plot title of graph
plt.title(f'PCA')
ax.legend(handles=scatter.legend_elements()[0],labels=labels)
plt.show()

print("end")