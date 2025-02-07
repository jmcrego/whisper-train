#This makes each class accessible when importing scripts as a package.
from .whisper import whisper
from .customIterableDataset import custom_iterable_dataset
from .computeMetrics import compute_metrics
from .dataCollators import DataCollatorSpeechSeq2SeqWithPadding

#The __all__ list explicitly defines what will be imported when using: from scripts import *
__all__ = ["whisper", "custom_iterable_dataset", "compute_metrics", "DataCollatorSpeechSeq2SeqWithPadding"]
