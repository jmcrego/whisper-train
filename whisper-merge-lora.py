#import json
import argparse
import logging
from peft import PeftModel, PeftConfig
from transformers import WhisperForConditionalGeneration, WhisperProcessor

def merge_lora(model_dir, out_dir=None):
    logging.info(f"Loading model with LoRA adapters from {model_dir}...")        
    # Load PEFT configuration to get the base model path
    logging.info(f"load {model_dir}: Peft model")        
    peft_config = PeftConfig.from_pretrained(model_dir)
    logging.info(f"load {model_dir}: base model: {peft_config.base_model_name_or_path}")        
    base_model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path)
    logging.info(f"load {model_dir}: processor")        
    processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path)

    # Load LoRA adapters into the base model
    model = PeftModel.from_pretrained(base_model, model_dir)
    logging.info("Merging LoRA adapters into the base model...")
    model = model.merge_and_unload()
    if out_dir is None:
        out_dir = model_dir+"_merged"
    logging.info(f"Saving merged model in {out_dir}...")
    model.save_pretrained(out_dir)
    processor.tokenizer.save_pretrained(out_dir)

    logging.info("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to merge LoRA adapters into base Whisper models.")
    parser.add_argument('model_dir', type=str, help='Path to LoRa params.')
    parser.add_argument('--odir', type=str, default=None, help='Output directory path (default [model_dir]_merged).')
    args = parser.parse_args()
    logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level='INFO', filename=None)

    merge_lora(args.model_dir, out_dir=args.odir)
