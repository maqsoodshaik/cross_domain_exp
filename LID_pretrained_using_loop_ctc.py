import torch
from torch.utils.data import DataLoader
import numpy as np
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer,Wav2Vec2Config,Wav2Vec2ForPreTraining,Wav2Vec2FeatureExtractor
from typing import Dict, List, Optional, Union
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
model_checkpoint = "facebook/wav2vec2-base"
from scipy.io.wavfile import write
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

dataset_name = "voxlingua"
from os import rename
from datasets import load_dataset, load_metric,concatenate_datasets,Dataset
import wandb
#set seed for reproducibility
def set_seed(seed):
    # Set the random seed manually for reproducibility.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_seed(42)
wandb.init(project='out_of_domain_pretrained')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = wandb.config
config.epochs = 50
config.batch_size = 2
configs = ['fr','de','nl','es','pl','pt','it']
list_datasets_train = []
list_datasets_validation = []
labels = configs
label2id, id2label,label2id_int = dict(), dict(),dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
    label2id_int[label] = i

num_labels = len(id2label)
# model = AutoModelForAudioClassification.from_pretrained(model_checkpoint, num_labels=num_labels,label2id=label2id,id2label=id2label)

# config_name = Wav2Vec2Config.from_pretrained(model_checkpoint)
config_name = Wav2Vec2Config()
config_name.output_hidden_states=True
ctc_model = Wav2Vec2ForPreTraining.from_pretrained(model_checkpoint,config=config_name)
# ctc_model = Wav2Vec2ForPreTraining(config=config_name)

for val,i in enumerate(configs):   
    print(val)
    dataset_train = load_dataset("/corpora/voxlingua/",data_dir= i,split = "train")
    dataset_train = Dataset.from_dict(dataset_train[:2000])
    dataset_train = dataset_train.add_column("labels",[val]*len(dataset_train))
    
    # write('output_voxlingua.wav', 16000, dataset_train[0]["audio"]["array"])
    dataset_validation = load_dataset("/corpora/voxlingua/",data_dir=i,split = "validation")
    dataset_validation = Dataset.from_dict(dataset_validation[:2000])
    dataset_validation = dataset_validation.add_column("labels",[val]*len(dataset_validation))
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



from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

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
    # inputs["labels"] = [label2id_int_f[image] for image in examples["label"]]
    # inputs["labels"] = examples["label"]
    return inputs

"""The feature extractor will return a list of numpy arays for each example:"""

# preprocess_function(dataset[:5])

"""To apply this function on all utterances in our dataset, we just use the `map` method of our `dataset` object we created earlier. This will apply the function on all the elements of all the splits in `dataset`, so our training, validation and testing data will be preprocessed in one single command."""

encoded_dataset_train = dataset_train.map(preprocess_function, remove_columns=["audio","label"], batched=True)
encoded_dataset_validation = dataset_validation.map(preprocess_function, remove_columns=["audio","label"], batched=True)
encoded_dataset_train.set_format("torch")
encoded_dataset_validation.set_format("torch")
data_collator = DataCollatorForWav2Vec2Pretraining(
        model=ctc_model, feature_extractor=feature_extractor
    )
train_dataloader = DataLoader(encoded_dataset_train, batch_size=config.batch_size,shuffle=True,drop_last=True,collate_fn=data_collator,)
eval_dataloader = DataLoader(encoded_dataset_validation, batch_size=config.batch_size,collate_fn=data_collator)
"""Even better, the results are automatically cached by the 🤗 Datasets library to avoid spending time on this step the next time you run your notebook. The 🤗 Datasets library is normally smart enough to detect when the function you pass to map has changed (and thus requires to not use the cache data). 🤗 Datasets warns you when it uses cached files, you can pass `load_from_cache_file=False` in the call to `map` to not use the cached files and force the preprocessing to be applied again.

### Training the model

Now that our data is ready, we can download the pretrained model and fine-tune it. For classification we use the `AutoModelForAudioClassification` class. Like with the feature extractor, the `from_pretrained` method will download and cache the model for us. As the label ids and the number of labels are dataset dependent, we pass `num_labels`, `label2id`, and `id2label` alongside the `model_checkpoint` here:
"""


"""The warning is telling us we are throwing away some weights (the `quantizer` and `project_q` layers) and randomly initializing some other (the `projector` and `classifier` layers). This is expected in this case, because we are removing the head used to pretrain the model on an unsupervised Vector Quantization objective and replacing it with a new head for which we don't have pretrained weights, so the library warns us we should fine-tune this model before using it for inference, which is exactly what we are going to do.

To instantiate a `Trainer`, we will need to define the training configuration and the evaluation metric. The most important is the [`TrainingArguments`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments), which is a class that contains all the attributes to customize the training. It requires one folder name, which will be used to save the checkpoints of the model, and all other arguments are optional:
"""
model_name_extension = "".join(configs)
model_name = model_checkpoint.split("/")[-1]+model_name_extension+dataset_name+"same_size"

# args = TrainingArguments(
#     f"/pretrained/{model_name}",#{model_name}arnlpt
#     evaluation_strategy = "epoch",
#     save_strategy = "epoch",
#     learning_rate=3e-5,
#     per_device_train_batch_size=config.batch_size,
#     gradient_accumulation_steps=4,
#     per_device_eval_batch_size=config.batch_size,
#     num_train_epochs=5,
#     warmup_ratio=0.1,
#     logging_steps=10,
#     load_best_model_at_end=True,
#     metric_for_best_model="accuracy",
# )
# print(f"hell:{args.no_cuda}")
"""Here we set the evaluation to be done at the end of each epoch, tweak the learning rate, use the `batch_size` defined at the top of the notebook and customize the number of epochs for training, as well as the weight decay. Since the best model might not be the one at the end of training, we ask the `Trainer` to load the best model it saved (according to `metric_name`) at the end of training.

The last argument `push_to_hub` allows the Trainer to push the model to the [Hub](https://huggingface.co/models) regularly during training. Remove it if you didn't follow the installation steps at the top of the notebook. If you want to save your model locally with a name that is different from the name of the repository, or if you want to push your model under an organization and not your name space, use the `hub_model_id` argument to set the repo name (it needs to be the full name, including your namespace: for instance `"anton-l/wav2vec2-finetuned-ks"` or `"huggingface/anton-l/wav2vec2-finetuned-ks"`).

Next, we need to define a function for how to compute the metrics from the predictions, which will just use the `metric` we loaded earlier. The only preprocessing we have to do is to take the argmax of our predicted logits:
"""
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear_1 = torch.nn.Linear(768, 256)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(256, num_labels)
    def forward(self, x):
        out = self.linear_1(x)
        out = self.relu(out)
        out = self.linear_2(out)
        return out
linear = LinearRegression()

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

"""You might wonder why we pass along the `feature_extractor` as a tokenizer when we already preprocessed our data. This is because we will use it once last time to make all the samples we gather the same length by applying padding, which requires knowing the model's preferences regarding padding (to the left or right? with which token?). The `feature_extractor` has a pad method that will do all of this for us, and the `Trainer` will use it. You can customize this part by defining and passing your own `data_collator` which will receive the samples like the dictionaries seen above and will need to return a dictionary of tensors.

Now we can finetune our model by calling the `train` method:
"""
from transformers import get_linear_schedule_with_warmup, AdamW
optimizer = AdamW(ctc_model.parameters(), lr=3e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader)*config.epochs*0.1, num_training_steps=len(train_dataloader)*config.epochs)
ctc_model = ctc_model.to(device)
criterion = torch.nn.CrossEntropyLoss()
linear.to(device)
#disable gradient calculation for the function
@torch.no_grad()
def eval_func(eval_dataloader, best_model):
    n_correct = 0
    n_samples = 0
    for batch in eval_dataloader:
        best_model.eval()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = best_model(input_values = batch['input_values'],mask_time_indices = batch['mask_time_indices'],sampled_negative_indices = batch['sampled_negative_indices'])
        out_logits = linear(outputs.hidden_states[-1])
        loss = criterion(out_logits.mean(dim=1), batch["labels"]) +outputs.loss
        _, predicted = torch.max(out_logits.mean(dim=1), 1)
        n_samples += 1
        n_correct += (predicted == batch["labels"]).sum().item()
    acc = 100.0 * n_correct / len(encoded_dataset_validation)
    print(f'Accuracy of the network on the validation images: {acc} %')
    #log to wandb
    wandb.log({f"Accuracy on validation of": acc})
    wandb.log({f"Loss on validation of": loss})
balance = 1.5
for epoch in range(1, config.epochs + 1):
    print(f"Epoch {epoch}")
    n_correct = 0
    n_samples = 0
    for batch_iter,batch in enumerate(train_dataloader):
        ctc_model.train()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = ctc_model(input_values = batch['input_values'],mask_time_indices = batch['mask_time_indices'],sampled_negative_indices = batch['sampled_negative_indices'])
        # outputs = ctc_model(batch["input_values"])
        out_logits = linear(outputs.hidden_states[-1])
        loss = criterion(out_logits.mean(dim=1), batch["labels"])+balance*outputs.loss
        print(f"loss in batch {batch_iter+1} is {loss}")
        #log loss using wandb
        wandb.log({"loss":loss})
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    with torch.no_grad():
        eval_func(eval_dataloader, ctc_model)
breakpoint()
print(trainer.train())


"""We can check with the `evaluate` method that our `Trainer` did reload the best model properly (if it was not the last one):"""

print(trainer.evaluate())

"""You can now upload the result of the training to the Hub, just execute this instruction:"""
trainer.save_model( f"/pretrained/{model_name}_bestmodel")
best_model = AutoModelForAudioClassification.from_pretrained(
    f"/pretrained/{model_name}_bestmodel"
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