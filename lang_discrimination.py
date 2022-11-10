from datasets import load_dataset, load_metric,concatenate_datasets,Dataset
from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
from torch.utils.data import DataLoader
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

metric = load_metric("accuracy")
model_checkpoint = "facebook/wav2vec2-base"
batch_size = 16
feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
max_duration = 10.0  # seconds
#fleurs
# dataset_name = "fleurs"
configs_v = ['fr_fr','de_de','nl_nl']
labels_v =["French","German","Dutch"]
label2id_v, id2label_v,label2id_int_v = dict(), dict(),dict()

for i, label in enumerate(labels_v):
    label2id_v[label] = str(i)
    id2label_v[str(i)] = label
    label2id_int_v[label] = i
list_datasets_validation_v = []
for i in configs_v:   
    dataset_validation_v = load_dataset("google/fleurs",i,split = "train")
    # dataset_validation = Dataset.from_dict(dataset_validation[:200])
    list_datasets_validation_v.append(dataset_validation_v)
dataset_validation_v = concatenate_datasets(
        list_datasets_validation_v
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
    inputs["labels"] = [label2id_int_v[image] for image in examples["language"]]
    return inputs
encoded_dataset_validation_v = dataset_validation_v.map(preprocess_function_f, remove_columns=["id","num_samples", "path", "audio", "transcription", "raw_transcription",  "lang_id", "language", "lang_group_id"], batched=True)

dataset_validation_combined_v= concatenate_datasets(
        [encoded_dataset_validation_v]
    )
dataset_name = "multilingual_librispeech"
from os import rename
from datasets import load_dataset, load_metric,concatenate_datasets
configs = ['french', 'german', 'dutch']
list_datasets_train = []
list_datasets_validation = []
labels = configs
label2id, id2label,label2id_int = dict(), dict(),dict()
best_model = AutoModelForAudioClassification.from_pretrained(
    model_checkpoint,output_hidden_states=True
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
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

trainer = Trainer(
    best_model,
    args,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics
)
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
    label2id_int[label] = i
for val,i in enumerate(configs):   
    dataset_train = load_dataset("facebook/multilingual_librispeech",i,split = "train.9h")
    dataset_train = dataset_train.add_column("labels",[val]*len(dataset_train))
    dataset_validation = load_dataset("facebook/multilingual_librispeech",i,split = "train.1h")
    dataset_validation = dataset_validation.add_column("labels",[val]*len(dataset_validation))
    list_datasets_train.append(dataset_train)
    list_datasets_validation.append(dataset_validation)
dataset_train = concatenate_datasets(
        list_datasets_train
    )
dataset_validation = concatenate_datasets(
        list_datasets_validation
    )

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * max_duration), 
        truncation=True,
        padding=True 
    )
    return inputs

"""The feature extractor will return a list of numpy arays for each example:"""

# preprocess_function(dataset[:5])

"""To apply this function on all utterances in our dataset, we just use the `map` method of our `dataset` object we created earlier. This will apply the function on all the elements of all the splits in `dataset`, so our training, validation and testing data will be preprocessed in one single command."""

encoded_dataset_train = dataset_train.map(preprocess_function, remove_columns=['file','audio','text','speaker_id','chapter_id','id'], batched=True)
encoded_dataset_validation = dataset_validation.map(preprocess_function, remove_columns=['file','audio','text','speaker_id','chapter_id','id'], batched=True)

dataset_validation_combined= concatenate_datasets(
        [encoded_dataset_train]
    )

best_model= best_model.wav2vec2

dataset_validation_combined.set_format("torch")
eval_dataloader = DataLoader(dataset_validation_combined, batch_size=16)
pred = torch.tensor([])
d = {}
f = {}
for x in range(13):
    d[f"hidden_state_{x}"] = torch.tensor([])
labels_p= torch.tensor([])
lang_p = torch.tensor([])
domain= torch.tensor([])
best_model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = best_model(batch["input_values"])
        
        for x in range(13):
            # breakpoint()
            f[f"hidden_state_s{x}"] = outputs.hidden_states[x].reshape(outputs.hidden_states[x].shape[0],-1)
        for x in range(13):
            
            d[f"hidden_state_{x}"] = torch.cat((d[f"hidden_state_{x}"],f[f"hidden_state_s{x}"].to("cpu")),0)
        
        pred_s = outputs.last_hidden_state.reshape(outputs.last_hidden_state.shape[0],-1).to("cpu")
        pred = torch.cat((pred,pred_s),0)
        # labels_s = batch["gender"].to("cpu")
        # labels_p = torch.cat((labels_p,labels_s),0)
        labels_s = batch["labels"].to("cpu")
        labels_p = torch.cat((labels_p,labels_s),0)
        # domain_s =  batch["domain"].to("cpu")
        # domain = torch.cat((domain,domain_s),0)
#calculating the accuracy of outof domain
dataset_validation_combined_v.set_format("torch")
eval_dataloader_v = DataLoader(dataset_validation_combined_v, batch_size=16)
for x in range(13):
    d[f"hidden_state_v_{x}"] = torch.tensor([])
labels_p_v= torch.tensor([])
best_model.eval()
for batch in eval_dataloader_v:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = best_model(batch["input_values"])
        
        for x in range(13):
            # breakpoint()
            f[f"hidden_state_s_v_{x}"] = outputs.hidden_states[x].reshape(outputs.hidden_states[x].shape[0],-1)
        for x in range(13):
            
            d[f"hidden_state_v_{x}"] = torch.cat((d[f"hidden_state_v_{x}"],f[f"hidden_state_s_v_{x}"].to("cpu")),0)
        
        # pred_s = outputs.last_hidden_state.reshape(outputs.last_hidden_state.shape[0],-1).to("cpu")
        # pred = torch.cat((pred,pred_s),0)
        # labels_s = batch["gender"].to("cpu")
        # labels_p = torch.cat((labels_p,labels_s),0)
        labels_s_v = batch["labels"].to("cpu")
        labels_p_v = torch.cat((labels_p_v,labels_s_v),0)
        # domain_s =  batch["domain"].to("cpu")
        # domain = torch.cat((domain,domain_s),0)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import random
def train_clf(x_train, x_test,y_train, y_test,accuracies,x_out_domain,y_out_domain,out_domain_accuracies):
    random.seed(0)
    np.random.seed(0)

    clf = LogisticRegression()


    clf.fit(x_train.reshape(x_train.shape[0],-1), y_train)
    score_test = clf.score(x_test.reshape(x_test.shape[0],-1), y_test)
    accuracies.append(score_test)
    score_train = clf.score(x_train.reshape(x_train.shape[0],-1), y_train)
    out_domain = clf.score(x_out_domain.reshape(x_out_domain.shape[0],-1), y_out_domain)
    out_domain_accuracies.append(out_domain)
    print(f"test_score{score_test}")
    print(f"train{score_train}")
    print(f"out_domain{out_domain}")
accuracies = [] 
out_domain_accuracies = []
for x in range(13):
    print(f"layer-{x+1}")
    X_train, X_test, y_train, y_test = train_test_split( d[f"hidden_state_{x}"], labels_p, test_size=0.33, random_state=42)
    train_clf( X_train, X_test, y_train, y_test,accuracies, d[f"hidden_state_v_{x}"], labels_p_v,out_domain_accuracies)
#plot accuracies against layers and save plot with name of the model_checkpoint used
import matplotlib.pyplot as plt
plt.plot(accuracies)
plt.xlabel("layers")
plt.ylabel("accuracy")
plt.savefig(f"/plots/lang_accuracies_{model_checkpoint.split('/')[-1]}_{dataset_name}.png")
#new plot for out_domain_accuracies
plt.plot(out_domain_accuracies)
plt.xlabel("layers")
plt.ylabel("accuracy")
plt.savefig(f"/plots/lang_out_domain_accuracies_{model_checkpoint.split('/')[-1]}_{dataset_name}.png")