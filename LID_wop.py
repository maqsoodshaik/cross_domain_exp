import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# -*- coding: utf-8 -*-

model_checkpoint = "facebook/wav2vec2-base"
batch_size = 8


dataset_name = "common_voice"
from os import rename
from datasets import load_dataset, load_metric,concatenate_datasets, Audio,Dataset
configs = ['fr','de','nl','es','it','pt','pl']
list_datasets_train = []
list_datasets_validation = []
for i in configs:   
    dataset_train = load_dataset("common_voice",i,split = "train")
    dataset_validation = load_dataset("common_voice",i,split = "validation")
    dataset_train = dataset_train.cast_column("audio", Audio(sampling_rate=16000))
    dataset_validation = dataset_validation.cast_column("audio", Audio(sampling_rate=16000))
    dataset_train = Dataset.from_dict(dataset_train[:25000])
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

from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
feature_extractor

"""As we've noticed earlier, the samples are roughly 1 second long, so let's set it here:"""

max_duration = 10.0  # seconds

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
model_name = model_checkpoint.split("/")[-1]+model_name_extension+dataset_name

args = TrainingArguments(
    f"/wop/{model_name}",#{model_name}arnlpt
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=30,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)
# print(f"hell:{args.no_cuda}")
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
trainer.save_model( f"/wop/{model_name}_bestmodel")
best_model = AutoModelForAudioClassification.from_pretrained(
    f"/wop/{model_name}_bestmodel"
)
trainer = Trainer(
    best_model,
    args,
    eval_dataset=encoded_dataset_validation,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics
)
print(f"after loading model:{trainer.evaluate()}")
# trainer.push_to_hub()

"""You can now share this model with all your friends, family, favorite pets: they can all load it with the identifier `"your-username/the-name-you-picked"` so for instance:

```python
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained("anton-l/my-awesome-model")
model = AutoModelForAudioClassification.from_pretrained("anton-l/my-awesome-model")

```
"""