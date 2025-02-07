import json
import argparse
from scripts import whisper

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to fine-tune Whisper models.")
    parser.add_argument('model', type=str, help='Path or name of the Whisper model fo fine-tune. Ex: openai/whisper-small')
    parser.add_argument('--train', nargs='+', default=[], required=True, help='Training map files.')    
    parser.add_argument('--valid', nargs='+', default=[], required=False, help='Validation map files.')
    parser.add_argument('--odir', type=str, required=True, help='Output directory path.')
    parser.add_argument('--language', type=str, default='french', help='Language of audio dataset.')
    parser.add_argument('--task', type=str, default='transcribe', help='Task to fine-tune.')
    parser.add_argument('--log', type=str, default='info', help='Logging level')
    parser.add_argument('--pars', type=str, default=None, help="JSON dictionary to control training. Ex: '{\"train_strategy\": \"steps\", \"eval_steps\": 1000}' (check the train function in whisperPeft:train() for allowed parameters)")
    parser.add_argument('--use_lora', action='store_true', help='Fine-tune Whisper with LoRA adaptors')
    args = parser.parse_args()

    w = whisper(model_name=args.model, language=args.language, task=args.task, log=args.log, use_lora=args.use_lora)
    pars = json.loads(args.pars) if args.pars is not None else {}
    w.train(args.train, args.valid, args.odir, **pars)