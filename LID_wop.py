import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# -*- coding: utf-8 -*-
"""Audio-Classification-on-Keyword-Spotting.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/audio_classification.ipynb

# **Fine-tuning for Audio Classification with 🤗 Transformers**

This notebook shows how to fine-tune multi-lingual pretrained speech models for Automatic Speech Recognition.

This notebook is built to run on the **Keyword Spotting** subset of the [SUPERB dataset](https://huggingface.co/datasets/superb) with any speech model checkpoint from the [Model Hub](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=downloads) as long as that model has a version with a Sequence Classification head (e.g. [Wav2Vec2ForSequenceClassification](https://huggingface.co/transformers/model_doc/wav2vec2.html#wav2vec2forsequenceclassification)). 

Depending on the model and the GPU you are using, you might need to adjust the batch size to avoid out-of-memory errors. Set those two parameters, then the rest of the notebook should run smoothly:
"""

model_checkpoint = "facebook/wav2vec2-base"
batch_size = 32

"""Before we start, let's install both `datasets` and `transformers` from master. Also, we need the `librosa` package to load audio files."""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install datasets==1.14
# !pip install transformers==4.11.3
# !pip install librosa

"""If you're opening this notebook locally, make sure your environment has an install from the last version of those libraries.

To be able to share your model with the community and generate results like the one shown in the picture below via the inference API, there are a few more steps to follow.

First you have to store your authentication token from the Hugging Face website (sign up [here](https://huggingface.co/join) if you haven't already!) then execute the following cell and input your username and password:
"""

# from huggingface_hub import notebook_login

# notebook_login()

"""
Then you need to install Git-LFS to upload your model checkpoints:"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !apt install git-lfs

"""## Fine-tuning a model on an audio classification task

In this notebook, we will see how to fine-tune one of the [🤗 Transformers](https://github.com/huggingface/transformers) acoustic models to a Keyword Spotting task of the [SUPERB Benchmark](https://superbbenchmark.org/)

Keyword Spotting (KS) detects preregistered keywords by classifying utterances into a predefined set of words. SUPERB uses the widely used Speech Commands dataset v1.0 for the task. The dataset consists of ten classes of keywords, a class for silence, and an unknown class to include the false positive.


### Loading the dataset

We will use the [🤗 Datasets](https://github.com/huggingface/datasets) library to download the data and get the Accuracy metric we need to use for evaluation. This can be easily done with the functions `load_dataset` and `load_metric`.
"""

from os import rename
from datasets import load_dataset, load_metric,concatenate_datasets
configs = ['uk','ab','pl','ru']
list_datasets_train = []
list_datasets_validation = []
for i in configs:   
    dataset_train = load_dataset("common_voice",i,split = "train")
    dataset_validation = load_dataset("common_voice",i,split = "validation")
    list_datasets_train.append(dataset_train)
    list_datasets_validation.append(dataset_validation)
dataset_train = concatenate_datasets(
        list_datasets_train
    )
dataset_validation = concatenate_datasets(
        list_datasets_validation
    )
metric = load_metric("accuracy")

"""The `dataset` object itself is a [`DatasetDict`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasetdict), which contains one key for the training, validation and test set."""

# print(dataset)

"""To access an actual element, you need to select a split first, then give an index:"""

# print(dataset["test"][1000])

"""As you can see, the `label` field is not an actual string label. By default the `ClassLabel` fields are encoded into integers for convenience:"""

# print(dataset["train"].features["label"])

"""Let's create an `id2label` dictionary to decode them back to strings and see what they are. The inverse `label2id` will be useful too, when we load the model later."""

labels = configs
label2id, id2label,label2id_int = dict(), dict(),dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
    label2id_int[label] = i
"""`Wav2Vec2` expects the input in the format of a 1-dimensional array of 16 kHz. This means that the audio file has to be loaded and resampled.

 Thankfully, `datasets` does this automatically when calling the column `audio`. Let try it out. 
"""

# dataset["test"][1000]["audio"]

"""We can see that the audio file has automatically been loaded. This is thanks to the new [`"Audio"` feature](https://huggingface.co/docs/datasets/package_reference/main_classes.html?highlight=audio#datasets.Audio) introduced in `datasets == 1.13.3`, which loads and resamples audio files on-the-fly upon calling.

The sampling rate is set to 16kHz which is what `Wav2Vec2` expects as an input.

To get a sense of what the commands sound like, the following snippet will render 
some audio examples picked randomly from the dataset. 

**Note**: *You can run the following cell a couple of times to listen to different audio samples.*
"""

# import random
# from IPython.display import Audio, display

# for _ in range(5):
#     rand_idx = random.randint(0, len(dataset["train"])-1)
#     example = dataset["train"][rand_idx]
#     audio = example["audio"]

#     print(f'Label: {id2label[str(example["label"])]}')
#     print(f'Shape: {audio["array"].shape}, sampling rate: {audio["sampling_rate"]}')
#     display(Audio(audio["array"], rate=audio["sampling_rate"]))
#     print()

"""If you run the cell a couple of times, you'll see that despite slight variations in length, most of the samples are about 1 second long (`duration = audio_length / sampling_rate`). So we can safely truncate and pad the samples to `16000`.

### Preprocessing the data

Before we can feed those audio clips to our model, we need to preprocess them. This is done by a 🤗 Transformers `FeatureExtractor` which will normalize the inputs and put them in a format the model expects, as well as generate the other inputs that the model requires.

To do all of this, we instantiate our feature extractor with the `AutoFeatureExtractor.from_pretrained` method, which will ensure that we get a preprocessor that corresponds to the model architecture we want to use.
"""

from transformers import AutoFeatureExtractor,Wav2Vec2Model

feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
feature_extractor

"""As we've noticed earlier, the samples are roughly 1 second long, so let's set it here:"""

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
    inputs["label"] = [label2id_int[image] for image in examples["locale"]]
    return inputs

"""The feature extractor will return a list of numpy arays for each example:"""

# preprocess_function(dataset[:5])

"""To apply this function on all utterances in our dataset, we just use the `map` method of our `dataset` object we created earlier. This will apply the function on all the elements of all the splits in `dataset`, so our training, validation and testing data will be preprocessed in one single command."""

encoded_dataset_train = dataset_train.map(preprocess_function, remove_columns=['locale','client_id', 'path', 'audio', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accent','segment'], batched=True)
encoded_dataset_validation = dataset_validation.map(preprocess_function, remove_columns=['locale','client_id', 'path', 'audio', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accent','segment'], batched=True)

# def transforms(examples):
#     examples["label"] = [label2id_int[image] for image in examples["locale"]]
#     return examples
# encoded_dataset_train.set_transform(transforms)
# encoded_dataset_validation.set_transform(transforms)
# encoded_dataset_train = encoded_dataset_train.map(remove_columns = ['locale'])
# encoded_dataset_validation = encoded_dataset_validation.map(remove_columns = ['locale'])
"""Even better, the results are automatically cached by the 🤗 Datasets library to avoid spending time on this step the next time you run your notebook. The 🤗 Datasets library is normally smart enough to detect when the function you pass to map has changed (and thus requires to not use the cache data). 🤗 Datasets warns you when it uses cached files, you can pass `load_from_cache_file=False` in the call to `map` to not use the cached files and force the preprocessing to be applied again.

### Training the model

Now that our data is ready, we can download the pretrained model and fine-tune it. For classification we use the `AutoModelForAudioClassification` class. Like with the feature extractor, the `from_pretrained` method will download and cache the model for us. As the label ids and the number of labels are dataset dependent, we pass `num_labels`, `label2id`, and `id2label` alongside the `model_checkpoint` here:
"""

from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer,Wav2Vec2ForSequenceClassification,Wav2Vec2Config


num_labels = len(id2label)



cnf = Wav2Vec2Config()
cnf.num_labels=num_labels
cnf.label2id=label2id
cnf.id2label=id2label
w_o_pretrain_model = Wav2Vec2ForSequenceClassification(cnf)

"""The warning is telling us we are throwing away some weights (the `quantizer` and `project_q` layers) and randomly initializing some other (the `projector` and `classifier` layers). This is expected in this case, because we are removing the head used to pretrain the model on an unsupervised Vector Quantization objective and replacing it with a new head for which we don't have pretrained weights, so the library warns us we should fine-tune this model before using it for inference, which is exactly what we are going to do.

To instantiate a `Trainer`, we will need to define the training configuration and the evaluation metric. The most important is the [`TrainingArguments`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments), which is a class that contains all the attributes to customize the training. It requires one folder name, which will be used to save the checkpoints of the model, and all other arguments are optional:
"""
model_name_extension = "".join(configs)
model_name = model_checkpoint.split("/")[-1]+model_name_extension

args = TrainingArguments(
    f"/wop/{model_name}",#{model_name}arnlpt
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
    metric_for_best_model="accuracy",
)

"""Here we set the evaluation to be done at the end of each epoch, tweak the learning rate, use the `batch_size` defined at the top of the notebook and customize the number of epochs for training, as well as the weight decay. Since the best model might not be the one at the end of training, we ask the `Trainer` to load the best model it saved (according to `metric_name`) at the end of training.

The last argument `push_to_hub` allows the Trainer to push the model to the [Hub](https://huggingface.co/models) regularly during training. Remove it if you didn't follow the installation steps at the top of the notebook. If you want to save your model locally with a name that is different from the name of the repository, or if you want to push your model under an organization and not your name space, use the `hub_model_id` argument to set the repo name (it needs to be the full name, including your namespace: for instance `"anton-l/wav2vec2-finetuned-ks"` or `"huggingface/anton-l/wav2vec2-finetuned-ks"`).

Next, we need to define a function for how to compute the metrics from the predictions, which will just use the `metric` we loaded earlier. The only preprocessing we have to do is to take the argmax of our predicted logits:
"""

import numpy as np

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

"""Then we just need to pass all of this along with our datasets to the `Trainer`:"""
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(encoded_dataset[:]['input_values'], encoded_dataset[:]['label'], test_size=0.33, random_state=42)
# encoded_dataset_train = {}
# encoded_dataset_train['input_values'] = X_train
# encoded_dataset_train['lable'] = y_train
# encoded_dataset_test = {}
# encoded_dataset_test['input_values'] = X_test
# encoded_dataset_test['lable'] = y_test
trainer = Trainer(
    w_o_pretrain_model,
    args,
    train_dataset=encoded_dataset_train,
    eval_dataset=encoded_dataset_validation,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics
)

"""You might wonder why we pass along the `feature_extractor` as a tokenizer when we already preprocessed our data. This is because we will use it once last time to make all the samples we gather the same length by applying padding, which requires knowing the model's preferences regarding padding (to the left or right? with which token?). The `feature_extractor` has a pad method that will do all of this for us, and the `Trainer` will use it. You can customize this part by defining and passing your own `data_collator` which will receive the samples like the dictionaries seen above and will need to return a dictionary of tensors.

Now we can finetune our model by calling the `train` method:
"""

print(trainer.train())

"""We can check with the `evaluate` method that our `Trainer` did reload the best model properly (if it was not the last one):"""

print(trainer.evaluate())

"""You can now upload the result of the training to the Hub, just execute this instruction:"""

# trainer.push_to_hub()

"""You can now share this model with all your friends, family, favorite pets: they can all load it with the identifier `"your-username/the-name-you-picked"` so for instance:

```python
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained("anton-l/my-awesome-model")
model = AutoModelForAudioClassification.from_pretrained("anton-l/my-awesome-model")

```
"""