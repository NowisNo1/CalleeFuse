from __future__ import unicode_literals, print_function, division

import os
import sys
import time
import argparse
import logging

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import torch
from torch.nn.utils import clip_grad_norm_

from torch.optim.adagrad import Adagrad

from training_ptr_gen.model import Model
from training_ptr_gen.eval import Evaluate
from data_util import config
from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util.utils import calc_running_avg_loss
from data_util.logging_util import (
    setup_logging, get_logger, TrainingLogger,
    format_duration, format_number, format_memory
)
from training_ptr_gen.train_util import get_input_from_batch, get_output_from_batch


use_cuda = config.use_gpu and torch.cuda.is_available()


def count_train_samples(data_path):
    """Count the training samples."""
    count = 0
    try:
        import subprocess
        result = subprocess.run(
            ['wc', '-l', data_path],
            capture_output=True, text=True, timeout=30
        )
        count = int(result.stdout.strip().split()[0])
    except Exception:
        # Fallback: count manually
        with open(data_path, 'r') as f:
            for _ in f:
                count += 1
    return count


def collect_config_dict():
    """Collect all hyperparameters into a dict for logging."""
    cfg = {}
    import data_util.config as c
    for attr in dir(c):
        if not attr.startswith('_') and attr.islower():
            val = getattr(c, attr)
            if not callable(val) and not isinstance(val, (type,)):
                cfg[attr] = val
    return cfg


class Train(object):
    def __init__(self, train_logger, train_dir):
        self.train_logger = train_logger
        self.train_dir = train_dir

        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
        time.sleep(1)

        self.model_dir = os.path.join(train_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.summary_writer = tf.summary.FileWriter(train_dir)

    def save_model(self, running_avg_loss, eval_loss, iter):
        """Save a separate checkpoint at every validation."""
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss,
            'eval_loss': eval_loss
        }
        model_save_path = os.path.join(self.model_dir, 'model_%d_%.4f' % (iter, eval_loss))
        torch.save(state, model_save_path)

        file_size = None
        try:
            file_size = os.path.getsize(model_save_path)
        except OSError:
            pass
        self.train_logger.log_save_model(iter, eval_loss, model_save_path, file_size)

    def setup_train(self, model_file_path=None):
        self.model = Model(model_file_path)

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        self.optimizer = Adagrad(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)

        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_iter, start_loss

    def train_one_batch(self, batch):
        enc_batch, enc_stmts, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)

        self.optimizer.zero_grad()
        encoder_outputs, encoder_feature, stmt_feature, encoder_hidden = self.model.encoder(enc_batch, enc_stmts, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(
                y_t_1, s_t_1,
                encoder_outputs, encoder_feature, stmt_feature, enc_padding_mask, c_t_1,
                extra_zeros, enc_batch_extend_vocab,
                coverage, di)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses / dec_lens_var
        loss = torch.mean(batch_avg_loss)

        loss.backward()

        encoder_norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        decoder_norm = clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        reduce_norm = clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)
        # Take the max of the three gradient norms as the overall gradient norm
        grad_norm = max(encoder_norm, decoder_norm, reduce_norm)

        self.optimizer.step()

        return loss.item(), grad_norm

    def trainIters(self, n_iters, model_file_path=None):
        iter, running_avg_loss = self.setup_train(model_file_path)
        logger = get_logger("train")

        # Print the training startup summary
        self.train_logger.log_train_start(collect_config_dict())

        step_start = time.time()
        # Used to compute the average time per log_interval
        interval_start = time.time()
        interval_step_count = 0

        # Track the lowest eval_loss (logging only, not for early stopping).
        # Note: no early stopping - in practice teacher-forcing eval_loss rises mid-training,
        #       while beam-search F1 keeps rising; loss-based early stopping would cut training before convergence.
        #       We therefore always run out max_iterations, keep all checkpoints, and let evaluate.py pick the best by F1.
        best_eval_loss = float('inf')
        best_eval_iter = 0

        while iter < n_iters:
            try:
                batch = self.batcher.next_batch()
                batch_start = time.time()
                raw_loss, grad_norm = self.train_one_batch(batch)
                batch_time = time.time() - batch_start

                running_avg_loss = calc_running_avg_loss(raw_loss, running_avg_loss, self.summary_writer, iter)
                iter += 1

                # ---- Per-step detailed log ----
                if iter % config.log_interval == 0:
                    self.train_logger.log_step(
                        step=iter,
                        raw_loss=raw_loss,
                        running_avg_loss=running_avg_loss,
                        grad_norm=grad_norm,
                        batch_time=batch_time,
                    )
                    self.summary_writer.flush()

                # ---- Validation + checkpoint saving ----
                if iter % config.eval_interval == 0:
                    self.train_logger.log_eval_start(iter)
                    eval_start = time.time()
                    eval_processor = Evaluate(model=self.model, train_dir=self.train_dir)
                    eval_loss, eval_steps = eval_processor.run_eval()
                    eval_time = time.time() - eval_start
                    self.train_logger.log_eval_end(iter, eval_loss, eval_steps, eval_time)

                    # Save at every validation (keep all checkpoints for post-training F1 selection)
                    self.save_model(running_avg_loss, eval_loss, iter)

                    # Only track the lowest eval_loss (no early stopping; always run max_iterations)
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        best_eval_iter = iter
                    logger.info(
                        "Step %s | eval_loss %.4f | lowest eval_loss so far %.4f @ step %s",
                        format_number(iter), eval_loss,
                        best_eval_loss, format_number(best_eval_iter))

                # ---- Speed monitoring (DEBUG-level file log) ----
                if iter % 10 == 0:
                    logger.debug(
                        "Step %s | raw_loss: %.4f | avg_loss: %.4f | grad_norm: %.2f | batch_time: %.3fs",
                        format_number(iter), raw_loss, running_avg_loss, grad_norm, batch_time
                    )

            except StopIteration:
                logger.warning(
                    "The training data generator is exhausted (all training files have been read), "
                    "but the current step %s has not reached the target %s; training stops early.",
                    format_number(iter), format_number(n_iters)
                )
                break

        # Training finished
        self.train_logger.log_train_end(iter, running_avg_loss)
        logger.info(
            "Training finished | ran %s/%s steps (always run out, no early stopping) | lowest eval_loss %.4f @ step %s | best checkpoint selected by evaluate.py by F1",
            format_number(iter), format_number(n_iters),
            best_eval_loss, format_number(best_eval_iter))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_file_path",
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    args = parser.parse_args()

    # ---- Initialize the training directory (logs, checkpoints, TensorBoard) ----
    train_dir = os.path.join(config.log_root, 'train_%d' % (int(time.time())))
    os.makedirs(train_dir, exist_ok=True)

    console_level = getattr(logging, config.console_log_level, logging.INFO)
    file_level = getattr(logging, config.file_log_level, logging.DEBUG)
    setup_logging(train_dir, log_name="training",
                  console_level=console_level, file_level=file_level)

    # ---- Count training data ----
    logger = get_logger("train")
    logger.info("Counting training samples...")
    train_sample_count = count_train_samples(config.train_data_path)
    logger.info("Training samples: %s", format_number(train_sample_count))

    # ---- Create the training logger ----
    train_logger = TrainingLogger(
        train_sample_count=train_sample_count,
        batch_size=config.batch_size,
        max_iterations=config.max_iterations,
        eval_data_path=config.eval_data_path,
        log_interval=config.log_interval,
        eval_interval=config.eval_interval,
    )

    # ---- Start training ----
    model_file_path = args.model_file_path
    train_processor = Train(train_logger, train_dir)
    train_processor.trainIters(config.max_iterations, model_file_path)