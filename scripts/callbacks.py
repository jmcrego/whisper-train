import time
import logging
from transformers import TrainerCallback

class WhisperCallback(TrainerCallback):
    def __init__(self):
        self.prev_step = 0
        self.prev_time = time.time()

    def on_evaluate(self, args, state, control, metrics=None, eval_dataloader=None, **kwargs):
        """Logs evaluation metrics after each evaluation step."""

        if metrics:
            logging.info(
                f"Evaluation | "
                f"Step: {state.global_step} | "
                f"epoch: {metrics.get('epoch', 0):.4f} | "
                f"loss: {metrics.get('eval_loss', 0):.4f} | "
                f"bleu: {metrics.get('eval_bleu', 0):.2f} | "
                f"wer: {metrics.get('eval_wer', 0):.2f} | "
                f"wer_ortho: {metrics.get('eval_wer_ortho', 0):.2f} | "
                f"runtime: {metrics.get('eval_runtime', 0):.2f} | "
                f"samples_per_second: {metrics.get('eval_samples_per_second', 0):.2f} | "
                f"steps_per_second: {metrics.get('eval_steps_per_second', 0):.4f}"
                #f"batch_size: {eval_dataloader.batch_size if eval_dataloader else 'N/A'}"
            )

    def on_log(self, args, state, control, logs=None, train_dataloader=None, **kwargs):
        """Logs training metrics at logging steps."""

        # Calculate steps per second = steps / elapsed_time
        curr_time = time.time()
        steps_per_second = (state.global_step - self.prev_step) / (curr_time - self.prev_time) if curr_time - self.prev_time > 0 else 0
        self.prev_step, self.prev_time = state.global_step, curr_time

        if logs:
            if logs.get('learning_rate', 0) == 0.:
                logging.info(
                    f"Training | "
                    f"Step: {state.global_step} no_optimize"
                )

                return
            # Log training metrics, ensure all keys are correctly referenced
            logging.info(
                f"Training | "
                f"Step: {state.global_step} | "
                f"epoch: {logs.get('epoch', 0):.4f} | "
                f"loss: {logs.get('loss', 0):.4f} | "
                f"learning_rate: {logs.get('learning_rate', 0):.8f} | "
                f"grad_norm: {logs.get('grad_norm', 0):.2f} | "
                f"steps_per_second: {steps_per_second:.4f} | "
                f"batch_size: {train_dataloader.batch_size if train_dataloader else 'N/A'}"
            )

    def on_save(self, args, state, control, **kwargs):
        processor_path = f"{args.output_dir}/checkpoint-{state.global_step}"        
        # Access the processor and tokenizer from the model
        model = state.trainer.model
        if hasattr(model, "processor"):
            model.processor.save_pretrained(processor_path)
            model.processor.tokenizer.save_pretrained(processor_path)
            print(f"Processor & Tokenizer saved at {processor_path}")
        else:
            print("Warning: Model does not have a processor attribute!")