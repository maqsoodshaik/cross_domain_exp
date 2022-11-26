import torch
model_checkpoint = "/wop/wav2vec2-basefrenchgermandutchspanishitalianportuguesepolishmultilingual_librispeech_bestmodel"#"/pretrained/wav2vec2-basefrenchgermandutchmultilingual_librispeech_bestmodel"#"/wop/wav2vec2-basefrenchgermandutchmultilingual_librispeech_bestmodel"
batch_size = 16
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
model_name_extension = "".join(configs_o)
model_name = model_checkpoint.split("/")[-1]+model_name_extension+dataset_name_o
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
# for batch in trainer.get_train_dataloader():
#     break
# from scipy.io.wavfile import write
# write('output_sounddevice.wav', 16000, np.array(batch["input_values"][0]))
# print(f"after loading model:{trainer.evaluate()}")
pred_o= trainer.predict(encoded_dataset_validation_o)
print(f"multilingual_accuracy:{pred_o}")
print(f"Multilingual_f1:{trainer_f1.predict(encoded_dataset_validation_o)}")





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
pred= trainer.predict(encoded_dataset_validation)
print(f"fleaurs_accuracy:{pred}")
print(f"fleaurs_f1:{trainer_f1.predict(encoded_dataset_validation)}")

#out of domain accuracy
encoded_dataset_validation_o=encoded_dataset_validation_o.add_column("domain",[0]*len(encoded_dataset_validation_o))
encoded_dataset_validation = encoded_dataset_validation.add_column("domain",[1]*len(encoded_dataset_validation))

dataset_validation_combined= concatenate_datasets(
        [encoded_dataset_validation,encoded_dataset_validation_o]
    )
best_model= best_model.wav2vec2
from torch.utils.data import DataLoader
dataset_validation_combined.set_format("torch")
eval_dataloader = DataLoader(dataset_validation_combined, batch_size=16)
pred = torch.tensor([])
labels_p= torch.tensor([])
domain= torch.tensor([])
best_model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = best_model(batch["input_values"])
        pred_s = outputs.last_hidden_state.reshape(outputs.last_hidden_state.shape[0],-1).to("cpu")
        pred = torch.cat((pred,pred_s),0)
        labels_s = batch["labels"].to("cpu")
        labels_p = torch.cat((labels_p,labels_s),0)
        domain_s =  batch["domain"].to("cpu")
        domain = torch.cat((domain,domain_s),0)
#calculating the accuracy of outof domain
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import random
def train_clf(x_train, x_test,y_train, y_test):
    random.seed(0)
    np.random.seed(0)

    clf = LogisticRegression()
    #clf = LogisticRegression(warm_start = True, penalty = 'l2',
    #                         solver = "saga", multi_class = 'multinomial', fit_intercept = False,
    #                         verbose = 5, n_jobs = 90, random_state = 1, max_iter = 15)
    #clf = SGDClassifier(max_iter=3000, n_iter_no_change=10)x
    #clf = LinearSVC(max_iter=3000)

    clf.fit(x_train.reshape(x_train.shape[0],-1), y_train)
    score_test = clf.score(x_test.reshape(x_test.shape[0],-1), y_test)
    score_train = clf.score(x_train.reshape(x_train.shape[0],-1), y_train)
    print(f"test_score{score_test}")
    print(f"train{score_train}")
X_train, X_test, y_train, y_test = train_test_split( pred, domain, test_size=0.33, random_state=42)
train_clf( X_train, X_test, y_train, y_test)
#plotting        
pred = pred.detach().numpy()
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# pred =pca.fit_transform(pred)
pred = TSNE(
        perplexity=6, n_iter=1000, learning_rate=200,random_state = 0
    ).fit_transform(pred)


import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots()
 
# Plot scaled features
xdata = pred[:,0]
ydata = pred[:,1]
import pandas as  pd
import seaborn as sns
labels__val = []
for i,l in enumerate(labels_p):
    if domain[i]==0:
        val=id2label[str(int(l))]
    else:
        val=id2label_f[str(int(l))]
    labels__val.append(val)
plot_frame= pd.DataFrame(list(zip(np.array(xdata.squeeze()),np.array(ydata.squeeze()),np.array(labels__val),np.array([id2domain[str(int(i))]for i in domain]))))
plot_frame.columns=["TSNE1","TSNE2","Labels","Domain"]
sns.scatterplot(x ="TSNE1" ,y="TSNE2",hue="Labels",style="Domain",data=plot_frame, alpha=0.6,marker='o')
# scatter =ax.scatter(xdata, ydata,s=np.array(domain),c=labels_p)
 
# Plot title of graph
plt.title(f'TSNE of original')
# ax.legend(handles=scatter.legend_elements()[0],labels=labels)
# ax.legend(handles=scatter.legend_elements()[1],labels=["indomain","outof_domain"])
plt.savefig(f"/plots/{model_checkpoint.split('/')[1]}_{dataset_name_o}_{''.join(labels)}_{dataset_name}_{''.join(labels_f)}.pdf", bbox_inches="tight")
plt.show()

print("end")

