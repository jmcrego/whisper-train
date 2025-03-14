import torch
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union
#from datasets import DatasetDict, Audio, load_from_disk, concatenate_datasets

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    #trainer: Any = field(default=None)

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features] # get the input features
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")# pad the input_features, it contains {'input_features': (bs, feat_dim, sl)}
        ### add labels to batch
        label_features = [{"input_ids": feature["labels"]} for feature in features] # get the tokenized label sequences
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt") # pad the labels
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100) # replace padding with -100 to ignore loss correctly
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item(): # if bos token is appended in previous tokenization step, cut bos token here as it's append later anyways
            labels = labels[:, 1:]
        batch["labels"] = labels
        #global_step = self.trainer.state.global_step+1 if self.trainer is not None else ''
        #logging.info(f'Step {global_step} batch: input_features.shape={list(batch["input_features"].shape)} (batch_size, mel_filters, frames) labels.shape={list(batch["labels"].shape)} (batch_size, seq_len)')
        return batch