from transformers import Wav2Vec2Processor, Data2VecVisionForImageClassification,Data2VecAudioForSequenceClassification,Data2VecVisionPreTrainedModel
from datasets import load_dataset
from transformers.modeling_outputs import BaseModelOutput,ImageClassifierOutput,BaseModelOutputWithPooling
import torch
from os import rename
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from datasets import load_dataset, load_metric,concatenate_datasets,Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
from transformers import AutoFeatureExtractor,Data2VecAudioConfig

labels =["French","German","Dutch"]
label2id, id2label,label2id_int = dict(), dict(),dict()

for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
    label2id_int[label] = i
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/data2vec-audio-base",return_tensors="pt")
max_duration = 10.0 
dataset_name = "fleurs"
configs = ['fr_fr','de_de','nl_nl']
list_datasets_validation = []
list_datasets_train = []
for i in configs:   
    dataset_train = load_dataset("google/fleurs",i,split = "train")
    # dataset_train = Dataset.from_dict(dataset_train[:20])
    dataset_validation = load_dataset("google/fleurs",i,split = "validation")
    # dataset_validation = Dataset.from_dict(dataset_validation[:20])
    list_datasets_train.append(dataset_train)
    list_datasets_validation.append(dataset_validation)
dataset_train = concatenate_datasets(
        list_datasets_train
    )
dataset_validation = concatenate_datasets(
        list_datasets_validation
    )
metric = load_metric("accuracy")
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
encoded_dataset_train = dataset_train.map(preprocess_function_f, remove_columns=["id","num_samples", "path", "audio", "transcription", "raw_transcription", "gender", "lang_id", "language", "lang_group_id"], batched=True)

encoded_dataset_validation = dataset_validation.map(preprocess_function_f, remove_columns=["id","num_samples", "path", "audio", "transcription", "raw_transcription", "gender", "lang_id", "language", "lang_group_id"], batched=True)
num_labels = len(id2label)
# load model and processor
model = Data2VecVisionForImageClassification.from_pretrained("facebook/data2vec-vision-base",num_labels=num_labels,
    label2id=label2id,
    id2label=id2label)
model_2 = Data2VecAudioForSequenceClassification.from_pretrained("facebook/data2vec-audio-base",num_labels=num_labels,
    label2id=label2id,
    id2label=id2label)
print(model)
print("##########")
print(model_2)
config = Data2VecAudioConfig.from_pretrained("facebook/data2vec-vision-base",num_labels=num_labels)
class CombinedModel(Data2VecVisionPreTrainedModel):

    def __init__(self,config,
        vision_model,audio_model
    ):
        """
        Args:
            data_dir (str): the path to the data on disk to read .npy files
            features_type (str): low-level speech features, e.g., MFCCs
            label_set (set): the set of labels (e.g., 'RUS', 'CZE', etc.)
            num_frames (int): the number of acoustic frames to sample from the
                speech signal, e.g., 300 frames is equivalent to 3 seconds
            max_num_frames (int): the max number of acoustic frames in input
                the diff. (max_num_frames - num_frames) is padded with zeros
            spectral_dim (int): the num of spectral components (default 13)
            start_index (int): the index of the 1st component (default: 0)
            end_index (int): the index of the last component (default: 13)
        """
        
        # get set of the labels in the dataset
        super().__init__(config)
        self.config = config
        self.relative_position_bias = None
        self.classifier = vision_model.classifier
        self.layer = vision_model.data2vec_vision.encoder.layer
        self.layernorm = vision_model.data2vec_vision.layernorm
        self.pooler = vision_model.data2vec_vision.pooler
        self.audio_model_feature_extractor = audio_model.data2vec_audio.feature_extractor
        self.feature_projection = audio_model.data2vec_audio.feature_projection
        self.pos_conv_embed = audio_model.data2vec_audio.encoder.pos_conv_embed
        self.layer_norm = audio_model.data2vec_audio.encoder.layer_norm
        self.dropout = audio_model.data2vec_audio.encoder.dropout
        self.gradient_checkpointing = False
        self.num_labels = config.num_labels
        vision_model.post_init()


    def forward(self, input_values,labels,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,head_mask=None,):
        ft_ext = self.audio_model_feature_extractor(input_values)
        ft_ext = ft_ext.transpose(1, 2)
        oudio_out = self.feature_projection(ft_ext)[0]
        pos = self.pos_conv_embed(oudio_out)
        hidden_states = oudio_out+pos
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    layer_head_mask,
                )
            else:
                relative_position_bias = (
                    self.relative_position_bias() if self.relative_position_bias is not None else None
                )
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions, relative_position_bias)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            encoder_outputs=  tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        else:
            encoder_outputs = BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]
        head_outputs = BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

        pooled_output = head_outputs.pooler_output if return_dict else head_outputs[1]

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + head_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=head_outputs.hidden_states,
            attentions=head_outputs.attentions,
        )
model=CombinedModel(config,model,model_2)
# model_2 = None
print(model)
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

num_labels = len(id2label)


model_name ="d2v_audio'"
batch_size=8
args = TrainingArguments(
    f"{model_name}",#{model_name}arnlpt
    evaluation_strategy = "steps",
    save_strategy = "steps",
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)
import numpy as np

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset_train,
    eval_dataset=encoded_dataset_validation,
    compute_metrics=compute_metrics,
)
print(trainer.train())