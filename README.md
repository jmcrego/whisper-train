# whisper-train
Python script to fine-tune Whisper models.

## Usage

### whisper-train.py

To fine-tune the original `whisper-small` model, using train/valid datasets indicated by the following map files and saving the adapted model into my-whisper-small, use:

```
$> python ./whisper-train.py openai/whisper-small \
  --train ../segments_trn.map \
  --valid ../segments_val.map \
  --odir my-whisper-small \
  --pars "{'train_strategy': 'steps', 'eval_steps': 1000}" 
```

Replace openai/whisper-small by your model if you want to fine tune a local checkpoint.
Use --pars to control training (allowed key/value pairs can be seen in train function of scripts/wisper:whisper class).

### segment_pakita.py
