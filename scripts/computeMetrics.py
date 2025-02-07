import os
import time
import logging
import evaluate

def save_file(fout, lout):
    with open(fout, 'w') as f:
        for out in lout:
            f.write(out + "\n")
    logging.info(f"Saved {fout}")

class compute_metrics():

    def __init__(self, processor, normalizer=None, trainer=None, save_dir=None):
        self.processor = processor
        self.normalizer = normalizer
        self.trainer = trainer
        self.save_dir = save_dir
        self.wer_metric = evaluate.load("wer", keep_in_memory=True)
        self.bleu_metric = evaluate.load("bleu", keep_in_memory=True)

    def __call__(self, pred):
        return self.whisper(pred)

    def whisper(self, pred):
        tic = time.time()

        pred_ids = pred.predictions
        label_ids = pred.label_ids
        logging.info(f'Scoring dataset with shape={pred_ids.shape}')

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        # Convert token IDs back to text
        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # compute scores
        scores = {}
        bleu = 100 * self.bleu_metric.compute(predictions=pred_str, references=[[ref] for ref in label_str])['bleu']
        scores["bleu"] = round(bleu,2)
        wer = 100 * self.wer_metric.compute(predictions=pred_str, references=label_str)
        scores["wer_ortho"] = round(wer,2)

        if self.normalizer is not None:
            # normalize predictions/references
            pred_str_norm = [self.normalizer(pred) for pred in pred_str]
            label_str_norm = [self.normalizer(label) for label in label_str]

            # discard empty references
            pred_str_norm = [pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i])]
            label_str_norm = [label_str_norm[i] for i in range(len(label_str_norm)) if len(label_str_norm[i])]

            # compute score
            wer = 100 * self.wer_metric.compute(predictions=pred_str_norm, references=label_str_norm)
            scores['wer'] = round(wer,2)

        if self.save_dir is not None:
            l = [f"{r.strip()}\t{p.strip()}" for r, p in zip(label_str, pred_str)]
            save_file(os.path.join(self.save_dir, f"ref_pred_{self.trainer.state.global_step}.txt"), l)        

        global_step = self.trainer.state.global_step if self.trainer is not None else 0
        logging.info(f'Step: {global_step} Scores: {scores} Took: {time.time()-tic:.2f} sec')
        return scores