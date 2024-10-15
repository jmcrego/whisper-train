import os
import fire
import torch
import logging
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import DatasetDict, Audio, load_from_disk, concatenate_datasets
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show ERROR and FATAL logs produced by tensorflow in the next imports
import evaluate
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback
from scripts.custom_iterable_dataset import custom_iterable_dataset
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper

def save_file(fout, lout):
    with open(fout, 'w') as f:
        for out in lout:
            f.write(out + "\n")
    logging.info(f"Saved {fout}")
    
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features] # get the input features
        label_features = [{"input_ids": feature["labels"]} for feature in features] # get the tokenized label sequences
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt") # pad the labels to max length
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100) # replace padding with -100 to ignore loss correctly
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item(): # if bos token is appended in previous tokenization step, cut bos token here as it's append later anyways
            labels = labels[:, 1:]
        batch["labels"] = labels
        logging.info(f'Step')
        #logging.info(f'Step: {self.trainer.state.global_step} batch input_features.shape={batch["input_features"].shape} labels.shape={batch["labels"].shape}')
        return batch
    
class whisper:
    
    def __init__(self, model_name='openai/whisper-medium', language='french', task='transcribe', log='info'):
        self.model_name = model_name
        self.language = language
        self.task = task
                
        logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=getattr(logging, log.upper()), filename=None)
        logging.getLogger('transformers').setLevel(logging.ERROR) #reduce transformer logging to error messages
        logging.getLogger('tensorflow').setLevel(logging.ERROR)

        if torch.cuda.is_available():
            logging.info(f"GPU is avalaible with {torch.cuda.device_count()} device(s)")
        else:
            logging.info("GPU is not avalaible")

        logging.info('############## MODEL LOADING... ##############')
    
        self.processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)    
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        
        if self.model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        self.normalizer = BasicTextNormalizer()
        self.metric = evaluate.load("wer")


    def compute_metrics(self, pred):
        logging.info('Scoring validation dataset')
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        # we do not want to group tokens when computing the metrics
        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        if self.normalize_eval:
            pred_str = [self.normalizer(pred) for pred in pred_str]
            label_str = [self.normalizer(label) for label in label_str]
        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)
        logging.info(f'Step: {self.trainer.state.global_step} WER: {wer}')
        save_file(os.path.join(self.output_dir, f"predictions_{self.trainer.state.global_step}.txt_wer_{wer}"), pred_str)        
        if self.trainer.state.global_step == self.eval_steps:
            save_file(os.path.join(self.output_dir, f"reference.txt"), label_str)
        return {"wer": wer}

    
    def train(
            self,
            train_datasets,
            eval_datasets,
            output_dir,
            train_strategy='steps',
            num_epochs = 1,
            eval_steps = 1000,
            save_steps = 1000,
            max_steps = 50000,
            warmup_steps = 100,
            batch_size = 32,
            learning_rate = 2.5e-5,
            gradient_accum = 1,
            lr_scheduler_type="constant_with_warmup",
            gradient_checkpointing=True,
            freeze_feature_encoder=False,
            freeze_encoder=False,
            normalize_eval = True,
    ):

        if train_strategy not in ['steps', 'epoch']:
            raise ValueError(f"--train_strategy should be steps or epoch, not {train_strategy}.")
    
        self.output_dir = output_dir
        self.normalize_eval = normalize_eval
        self.eval_steps = eval_steps
        
        if freeze_feature_encoder:
            self.model.freeze_feature_encoder()

        if freeze_encoder:
            self.model.freeze_encoder()
            self.model.model.encoder.gradient_checkpointing = False

        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []

        if gradient_checkpointing:
            self.model.config.use_cache = False            
            
        logging.info('############## DATASET LOADING... ##############')
    
        min_label_length = 5 + 3
        max_label_length = self.model.config.max_length
        min_duration = 0.0
        max_duration = 30.0
        ds_train = custom_iterable_dataset(train_datasets, language=self.language, sr=16000, mind=min_duration, maxd=max_duration, minl=min_label_length, maxl=max_label_length, clean=True, seed=None, processor=self.processor)
        ds_eval  = custom_iterable_dataset(eval_datasets,  language=self.language, sr=16000, mind=min_duration, maxd=max_duration, minl=min_label_length, maxl=max_label_length, clean=True, seed=None, processor=self.processor)
    
        logging.info('############## TRAINING... ##############')

        if train_strategy == 'epoch':
            p = {"eval_strategy": "epoch", "save_strategy": "epoch", "num_train_epochs": num_epochs}
        else:
            p = {"eval_strategy": "steps", "eval_steps": eval_steps, "save_strategy": "steps", "save_steps": save_steps, "max_steps": max_steps}
            
        training_args = Seq2SeqTrainingArguments(
            **p,
            output_dir=self.output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accum,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            warmup_steps=warmup_steps,
            gradient_checkpointing=gradient_checkpointing, #use model.gradient_checkpointing_enable() instead of this option
            fp16=True,
            save_total_limit=10,
            per_device_eval_batch_size=batch_size,
            predict_with_generate=True,
            generation_max_length=225,
            logging_steps=50,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            optim="adamw_bnb_8bit",
            resume_from_checkpoint=None,
        )

        self.trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            train_dataset=IterableWrapper(ds_train), #IterableWrapper is needed to avoid the lack of __len__ on iterabledataset
            eval_dataset=IterableWrapper(ds_eval),
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )
        
        self.trainer.train()
        logging.info('############## DONE ##############')




    
    
    
