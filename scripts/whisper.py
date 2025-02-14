import os
import time
import torch
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show ERROR and FATAL logs produced by tensorflow in the next imports
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback, GenerationConfig, BitsAndBytesConfig
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
#from torchdata.datapipes.iter import IterDataPipe#, IterableWrapper ###needs torchdata <0.10.0 i used: pip install torchdata==0.9.0
from peft import LoraConfig, get_peft_model

# to import the required classes from the same package (scripts), use relative/absolute imports (thanks to __init__.py).
from scripts.customIterableDataset import custom_iterable_dataset
from scripts.computeMetrics import compute_metrics
from scripts.dataCollators import DataCollatorSpeechSeq2SeqWithPadding
from scripts.callbacks import WhisperCallback

#class customSeq2SeqTrainer(Seq2SeqTrainer):
#    #def save_model(self, output_dir=None): ### rewrite the save_model function of Seq2SeqTrainer
#    def save_model(self, output_dir, _internal_call=False):
#        ### Save the model
#        super().save_model(output_dir, _internal_call=_internal_call)
#        logging.info(f"Model saved to {output_dir}") #it saves the tokenizer and feature extractor
#        ### Save the tokenizer too
#        if output_dir is not None: 
#            self.processor.save_pretrained(output_dir)
#            logging.info(f"Preprocessor saved to {output_dir}") #it saves the tokenizer and feature_extractor


class whisper:
    
    def __init__(self, model_name='openai/whisper-medium', language='french', task='transcribe', log='info', use_lora=False):
#        logging.info(f"whisper.init: {{key: value for key, value in locals().items() if key != 'self'}}")
        self.model_name = model_name
        self.language = language
        self.task = task
        self.use_lora = use_lora
        logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=getattr(logging, log.upper()), filename=None)
        logging.getLogger('transformers').setLevel(logging.ERROR) #reduce transformer logging to error messages
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        logging.info(f"GPU is avalaible with {torch.cuda.device_count()} device(s)" if torch.cuda.is_available() else "GPU is not avalaible")

        gradient_checkpointing=True
        freeze_feature_encoder=False
        freeze_encoder=False
        freeze_decoder=False
        # Manually clear GPU memory
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        logging.info('############## MODEL LOADING... ##############')
    
        if self.use_lora:
            bnb_config = None #BitsAndBytesConfig(load_in_8bit=True)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name, quantization_config=bnb_config, device_map="cuda")
            lora_config = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.1, bias="none", target_modules=["q_proj", "v_proj"]) # Define LoRA configuration
            self.model = get_peft_model(self.model, lora_config) # Wrap model with PEFT (only LoRA layers are trainable)
            self.model.print_trainable_parameters()  # Show trainable params

        else:
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name, device_map="cuda")

        self.processor = WhisperProcessor.from_pretrained(self.model_name, language=self.language, task=self.task)    
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        self.normalizer = BasicTextNormalizer()
        self.compute_metrics = compute_metrics(self.processor, normalizer=self.normalizer, trainer=None, save_dir=None)

        #Explicitly Set Language & Task in Model Configuration        
        self.model.config.forced_decoder_ids = self.processor.tokenizer.get_decoder_prompt_ids(language=self.language, task=self.task)
        self.model.config.suppress_tokens = []
        self.model.config.decoder_start_token_id = self.processor.tokenizer.bos_token_id
    
        # Disable caching (needed for training)
        self.model.config.use_cache = False 

        if freeze_feature_encoder:
            self.model.freeze_feature_encoder() ### already freezed if LoRA

        if freeze_encoder:
            self.model.freeze_encoder() ### already freezed if LoRA

        if freeze_decoder:
            #attention! freezing the decoder might stop LoRA layers from updating. You better freeze parts of the decoder
            for param in self.model.model.decoder.parameters():
                param.requires_grad = False  # Freeze all decoder parameters            

        if gradient_checkpointing:
            # Gradient checkpointing (optional). It saves memory by recomputing activations during the backward pass (with LoRA not really needed since few parameters). 
            # However, with a quantized model (like 8-bit or 4-bit quantization with bitsandbytes), gradient checkpointing can still be useful.
            self.model.gradient_checkpointing_enable()  # If memory is an issue

        # Ensure generation_config is set correctly, and set relevant parameters in the generation config (only needed for inference "generation" phase)
        #if self.model.generation_config is None:
        #    self.model.generation_config = GenerationConfig()
        #self.model.generation_config.language = self.language
        #self.model.generation_config.task = self.task
        #self.model.generation_config.forced_decoder_ids = self.model.config.forced_decoder_ids
        #self.model.generation_config.suppress_tokens = self.model.config.suppress_tokens

        logging.info(self.model.config)

    def train(
        self,
        train_datasets,
        eval_datasets,
        output_dir,
        logging_steps = 5,
        eval_steps = 50,
        save_steps = 50,
        max_steps = 5000,
        warmup_steps = 100,
        batch_size = 32,
        learning_rate = 5e-6,
        gradient_accum = 1,
        lr_scheduler_type="linear_with_warmup",
        resume_from=None,
        seed=None,
    ):
#        logging.info(f"whisper.train: {{key: value for key, value in locals().items() if key != 'self'}}")
        config_dir = os.path.join(output_dir, "config")
        self.processor.save_pretrained(config_dir) # Save processor
        self.processor.tokenizer.save_pretrained(config_dir) # Save tokenizer

        logging.info('############## DATASET LOADING... ##############')
    
        min_label_length = None #5 + 3
        max_label_length = None #self.model.config.max_length
        min_duration = None #0.0
        max_duration = None #30.0
        ds_train = custom_iterable_dataset(train_datasets, language=self.language, sr=16000, mind=min_duration, maxd=max_duration, minl=min_label_length, maxl=max_label_length, clean=False, seed=seed, processor=self.processor, iterable=True)
        ds_eval  = custom_iterable_dataset(eval_datasets,  language=self.language, sr=16000, mind=min_duration, maxd=max_duration, minl=min_label_length, maxl=max_label_length, clean=False, seed=None, processor=self.processor, iterable=False)

        logging.info(f"Training dataset size: {len(ds_train)}")
        logging.info(f"Batch size: {batch_size}")
        steps_per_epoch = len(ds_train) // (batch_size * gradient_accum)
        total_epochs = max_steps / steps_per_epoch
        logging.info(f"Expected total epochs: {total_epochs}")
        logging.info(f"Max Steps: {max_steps}")

        logging.info('############## TRAINING... ##############')
            
        ### https://huggingface.co/docs/transformers/main/main_classes/trainer#transformers.Seq2SeqTrainingArguments
        training_args = Seq2SeqTrainingArguments( 
            eval_strategy="steps",
            save_strategy="steps", 
            logging_strategy="steps", 
            eval_steps=eval_steps,
            save_steps=save_steps, 
            logging_steps=logging_steps,
            max_steps=max_steps,
            num_train_epochs=99,  # This seems to be needed to disalow stopping
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accum,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            warmup_steps=warmup_steps,
            optim="adamw_bnb_8bit",
            fp16=True,
            per_device_eval_batch_size=batch_size//2, # reduce memory needs
### next are for generation over the validation set
            predict_with_generate=True,
            generation_max_length=225,
            generation_num_beams=1,
### evaluation and saving
            report_to=["tensorboard"],
            load_best_model_at_end=True, # best checkpoint will always be saved
            metric_for_best_model="wer",
            greater_is_better=False,
            save_total_limit=2,
            eval_on_start=True,
        )

        ### https://huggingface.co/docs/transformers/main/main_classes/trainer#transformers.Seq2SeqTrainer
        callbacks = WhisperCallback()
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            train_dataset=ds_train,
            eval_dataset=ds_eval,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            processing_class=self.processor.feature_extractor,
            callbacks=[callbacks]
        )

        remaining_steps = training_args.max_steps - trainer.state.global_step
        trainer.args.max_steps = remaining_steps
        logging.info(f"Max Steps in Trainer: {trainer.args.max_steps}")
        logging.info(f"Num Train Epochs in Trainer: {trainer.args.num_train_epochs}")

        # Inject trainer/output_dir into compute_metrics to allow access to trainer.state.global_step and to save refs/hyps
        self.compute_metrics.trainer = trainer
        self.compute_metrics.save_dir = output_dir + '/eval'
        # Inject trainer into callbacks
        callbacks.processor = self.processor

        logging.info(f'Resume training from {resume_from}')
        trainer.train(resume_from_checkpoint=resume_from)

        if self.use_lora:
            self.model = self.model.merge_and_unload() # Merges LoRA weights into original model
            self.model.save_pretrained(output_dir+"_merged")  # Save the new model
            self.processor.save_pretrained(output_dir+"_merged") # Save processor
            self.processor.tokenizer.save_pretrained(output_dir+"_merged") # Save tokenizer

        logging.info('############## DONE ##############')




    
    
    
