# whisper-train
Python script to fine-tune Whisper models.

## Usage

### whisper-train.py

Use:
```
$> python ./whisper-train.py openai/whisper-small \
  --train ../segments_trn.map \
  --valid ../segments_val.map \
  --odir my-whisper-small 
```

### segment_pakita.py
