from datasets import load_dataset, load_metric,concatenate_datasets,Dataset
from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
from torch.utils.data import DataLoader
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

metric = load_metric("accuracy")
model_checkpoint = "/wop/wav2vec2-basefrenchgermandutchmultilingual_librispeech_bestmodel"
batch_size = 16
feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
max_duration = 10.0  # seconds

dataset_name = "fleurs"
configs = ['fr_fr','de_de','nl_nl']
labels =["French","German","Dutch"]
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
list_datasets_validation = []
for i in configs:   
    dataset_validation = load_dataset("google/fleurs",i,split = "train")
    # dataset_validation = Dataset.from_dict(dataset_validation[:200])
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
    inputs["labels"] = [label2id_int[image] for image in examples["language"]]
    return inputs
encoded_dataset_validation = dataset_validation.map(preprocess_function_f, remove_columns=["id","num_samples", "path", "audio", "transcription", "raw_transcription",  "lang_id", "language", "lang_group_id"], batched=True)

dataset_validation_combined= concatenate_datasets(
        [encoded_dataset_validation]
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
domain= torch.tensor([])
best_model.eval()
breakpoint()
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
        labels_s = batch["gender"].to("cpu")
        labels_p = torch.cat((labels_p,labels_s),0)
        # domain_s =  batch["domain"].to("cpu")
        # domain = torch.cat((domain,domain_s),0)
#calculating the accuracy of outof domain
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import random
def train_clf(x_train, x_test,y_train, y_test,accuracies):
    random.seed(0)
    np.random.seed(0)

    clf = LogisticRegression()


    clf.fit(x_train.reshape(x_train.shape[0],-1), y_train)
    score_test = clf.score(x_test.reshape(x_test.shape[0],-1), y_test)
    accuracies.append(score_test)
    score_train = clf.score(x_train.reshape(x_train.shape[0],-1), y_train)
    print(f"test_score{score_test}")
    print(f"train{score_train}")
accuracies = [] 
for x in range(13):
    print(f"layer-{x+1}")
    X_train, X_test, y_train, y_test = train_test_split( d[f"hidden_state_{x}"], labels_p, test_size=0.33, random_state=42)
    train_clf( X_train, X_test, y_train, y_test,accuracies)
#plot accuracies against layers and save plot with name of the model_checkpoint used
import matplotlib.pyplot as plt
plt.plot(accuracies)
plt.xlabel("layers")
plt.ylabel("accuracy")
print("saving plot")
plt.savefig(f"/plots/gender_accuracies_{model_checkpoint.split('/')[1]}_{model_checkpoint.split('/')[-1]}_{dataset_name}.png")