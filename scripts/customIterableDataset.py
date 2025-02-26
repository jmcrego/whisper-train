import re
import random
import argparse
import logging
from functools import partial
from datasets import Dataset, Audio#, IterableDataset #, Value
from torchdata.datapipes.iter import IterableWrapper ###needs torchdata <0.10.0 i used: pip install torchdata==0.9.0

noise = r"\[[^\s\]]+\]"
separ = r"\|\|\|"
hesit = r"\b(euh|beuh|hm|hein)\b"
punct = r"(\w)\s+([,.!?;:])"
apost = r"(\w')\s+(\w)"
chars = r"[\(\)\*&]"
initp = r"^\s*[,.!?;:]\s+"
spaces = r"\s\s+"

class IterableWrapperWithLen(IterableWrapper):
    def __init__(self, dataset, length):
        super().__init__(dataset)  # Initialize parent class
        self.length = length  # Manually specify length

    def __len__(self):
        return self.length  # Define the length explicitly

def clean_sentence(e):
    txt = e['sentence']
    txt = re.sub(noise, '', txt) #remove noises    
    #txt = re.sub(separ, ' ', txt) #remove segments separation
    #txt = re.sub(hesit, '', txt) #remove hesitations
    #txt = re.sub(punct, r'\1\2', txt) #join punctuation to preceding word: word ? => word?    
    #txt = re.sub(apost, r'\1\2', txt) #join apostrophes to next word: word' word => word'word
    txt = re.sub(chars, '', txt) #remove some chars    
    #txt = re.sub(initp, '', txt) #remove initial punctuation
    txt = re.sub(spaces, ' ', txt) #remove multiple spaces
    e['sentence'] = txt.strip()
    return e

def filter_by_duration(e, mind, maxd):
    if mind is not None or maxd is not None:
        duration_sec = len(e['audio']['array']) / e['audio']['sampling_rate']
        if mind is not None and duration_sec < mind:
            return False
        if maxd is not None and duration_sec > maxd:
            return False
    return True #keep

def filter_by_length(e, minl, maxl):
    if minl is not None or maxl is not None:
        length_tokens = len(e['labels'])
        if minl is not None and length_tokens < minl:
            return False
        if maxl is not None and length_tokens > maxl:
            return False
    return True #keep

def add_input_features_and_label_ids(p):
    def process_example(e):
        e["input_features"] = p.feature_extractor(e['audio']["array"], sampling_rate=e['audio']["sampling_rate"]).input_features[0]
        e["labels"] = p.tokenizer(e['sentence']).input_ids
        return e
    return process_example

def custom_iterable_dataset(files, language="french", sr=16000, mind=None, maxd=None, minl=None, maxl=None, clean=False, seed=None, processor=None, firstn=0, iterable=True):
    #logging.info(f"custom_iterable_dataset: {locals()}")
    paths, sentences, languages = [], [], []
    for i in range(len(files)):
        logging.info(f'read {files[i]}')
        with open(files[i],'r') as fd:
            for l in fd:
                toks = l.strip().split('\t')
                path, sentence = toks[0], toks[1]
                paths.append(path)
                sentences.append(sentence)
                languages.append(language)
                if firstn:
                    if len(paths) == firstn:
                        logging.info(f'dataset: use firstn={firstn} examples')
                        break

    if seed is not None:
        logging.info(f'dataset: shuffling {len(paths)} examples')                
        random.seed(seed)
        combined = sorted(zip(paths, sentences)) # Zip the two lists together and sort them based on the first list
        random.shuffle(combined) # Shuffle with the seed
        paths, sentences = zip(*combined) # Unzip the sorted pairs back into two separate lists
                
    logging.info(f'dataset: found {len(paths)} examples')                
    ds = Dataset.from_dict({"audio": paths, 'sentence': sentences, 'language': languages})

    logging.info(f'dataset: cast Audio(sampling_rate={sr})')
    ds = ds.cast_column("audio", Audio(sampling_rate=sr))

    if iterable:
        len_ds = len(ds)
        ds = ds.to_iterable_dataset()    
    
    if clean:
        logging.info('dataset: clean sentence')
        ds = ds.map(clean_sentence)
        
    if mind is not None or maxd is not None:
        logging.info(f'dataset: filter by {mind}<=duration<={maxd}')
        filter_func = partial(filter_by_duration, mind=mind, maxd=maxd)
        ds = ds.filter(filter_func)
        
    if processor is not None:
        logging.info('dataset: add input_features/label_ids')
        ds = ds.map(add_input_features_and_label_ids(processor))
        
    if minl is not None or maxl is not None:
        logging.info(f'dataset: filter by {minl}<=length<={maxl}')
        filter_func = partial(filter_by_length, minl=minl, maxl=maxl)
        ds = ds.filter(filter_func)

    if iterable:
        logging.info(f'dataset: iterable dataset (len={len_ds})')
        return IterableWrapperWithLen(ds, len_ds) 
    else:
        logging.info(f'dataset: kept {len(ds)} examples after filtering')
        return ds


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Load iterable dataset from file.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('map', type=str, nargs='+', default=[], help='List of map files containing audio examples (path, duration, sentence).')
    parser.add_argument('--model', type=str, default='openai/whisper-medium', help='Path or name of the Whisper model with processor.')
    parser.add_argument('--o', type=str, required=False, help='Output directory for dataset.')
    parser.add_argument('--l', type=str, required=False, default='french', help='Language of audios.')
    parser.add_argument('--sr', type=int, required=False, default=16000, help='Sampling rate of audios.')
    parser.add_argument('--clean', action='store_true', help='Clean sentence using regex rules.')
    parser.add_argument('--mind', type=float, required=False, default=None, help='Minimum audio duration in seconds.')
    parser.add_argument('--maxd', type=float, required=False, default=None, help='Maximum audio duration in seconds.')
    parser.add_argument('--minl', type=int, required=False, default=None, help='Minimum sentence length in ids. (lists contain at least 5 tokens: [\'<|startoftranscript|>\', \'<|fr|>\', \'<|transcribe|>\', \'<|notimestamps|>\', ..., \'<|endoftext|>\')')
    parser.add_argument('--maxl', type=int, required=False, default=None, help='Maximum sentence length in ids. (lists contain at least 5 tokens: [\'<|startoftranscript|>\', \'<|fr|>\', \'<|transcribe|>\', \'<|notimestamps|>\', ..., \'<|endoftext|>\')')
    parser.add_argument('--seed', type=int, required=False, default=None, help='Shuffle seed for dataset.')
    args = parser.parse_args()

    from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
    processor = WhisperProcessor.from_pretrained(args.model, language='french', task="transcribe")

    ds = custom_iterable_dataset(args.map, language=args.l, sr=args.sr, mind=args.mind, maxd=args.maxd, minl=args.minl, maxl=args.maxl, clean=args.clean, seed=args.seed, processor=processor)
    
    if args.o:
        ds.save_to_disk(args.o)
    print('begin iteration')
    for i,e in enumerate(ds):
        print(f'{e["audio"]["path"]}\t{e["sentence"]}\t{e["labels"]}')
