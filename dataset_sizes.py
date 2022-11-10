dataset_name = "multilingual_librispeech"
from os import rename
from datasets import load_dataset, load_metric,concatenate_datasets
configs = ['french', 'german', 'dutch']
list_datasets_train = []
list_datasets_validation = []
for val,i in enumerate(configs):   
    dataset_train = load_dataset("facebook/multilingual_librispeech",i,split = "train.9h")
    dataset_train = dataset_train.add_column("labels",[val]*len(dataset_train))
    list_datasets_train.append(dataset_train)
dataset_train = concatenate_datasets(
        list_datasets_train
    )
    
print(f"length of {dataset_name} : {len(dataset_train)}")

dataset_name = "voxlingua"

configs = ['fr','de','nl']
list_datasets_train = []
list_datasets_validation = []
for val,i in enumerate(configs):   
    dataset_train = load_dataset("/corpora/voxlingua/",data_dir= i,split = "train")
    dataset_train = dataset_train.add_column("labels",[val]*len(dataset_train))

    list_datasets_train.append(dataset_train)

dataset_train = concatenate_datasets(
        list_datasets_train
    )
print(f"length of {dataset_name} : {len(dataset_train)}")
dataset_name = "fleurs"
configs = ['fr_fr','de_de','nl_nl']
list_datasets_train = []
list_datasets_validation = []
for i in configs:   
    dataset_train = load_dataset("google/fleurs",i,split = "train")
    list_datasets_train.append(dataset_train)
dataset_train = concatenate_datasets(
        list_datasets_train
    )
print(f"length of {dataset_name} : {len(dataset_train)}")
from pathlib import Path
import datasets
datasets.config.DOWNLOADED_DATASETS_PATH = Path('/corpora/common_voice_speech/')
dataset_name = "common_voice"

configs = ['fr','de','nl']
list_datasets_train = []
list_datasets_validation = []
for i in configs:   
    dataset_train = load_dataset("common_voice",i,split = "train")
    list_datasets_train.append(dataset_train)
dataset_train = concatenate_datasets(
        list_datasets_train
    )
print(f"length of {dataset_name} : {len(dataset_train)}")

music = load_dataset("krandiash/beethoven","clean")
breakpoint()
print(f"length of music: {len(music)}")