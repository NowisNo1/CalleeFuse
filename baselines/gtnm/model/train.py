# -*- coding: utf-8 -*-
#/usr/bin/python3

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.disable_eager_execution()
# from new_data_loader import *
from model.model_invoked import Transformer
from data_processing.extract_data_subword import *
from model.utils import save_hparams, save_variable_specs, get_hypotheses, calc_bleu
import os
from model.hparams import Hparams
import math
import logging
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)


def _ts():
    """Current timestamp string for logging."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _fmt_dur(seconds):
    """Format seconds as hh:mm:ss."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return "%02d:%02d:%02d" % (h, m, s)

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
save_hparams(hp, hp.logdir)
os.environ['CUDA_VISIBLE_DEVICES'] = hp.gpu
outfile = hp.res_log

def run_epoch(session, model, state, summary_writer, epoch=None):
    total_loss = 0.0

    data_loader = model.data.batch_iter(hp.batch_size, state, epoch=epoch)
    step = 0
    while True:
        feed_dict = {}

        body_batch, pro_batch, doc_batch, dec_inp_batch, dec_tgt_batch, invoked_batch, batch_len = next(data_loader)
        feed_dict[model.body_batch] = body_batch
        feed_dict[model.pro_batch] = pro_batch
        feed_dict[model.doc_batch] = doc_batch
        feed_dict[model.invoked_batch] = invoked_batch
        feed_dict[model.dec_inp_batch] = dec_inp_batch
        feed_dict[model.dec_tgt_batch] = dec_tgt_batch

        _, _gs, _summary, _loss, _preds = session.run([model.train_op, model.global_step, model.train_summaries, model.loss, model.preds], feed_dict)
        summary_writer.add_summary(_summary, _gs)

        if step % (batch_len // 10) == 10:
            print("%.2f perplexity : %.3f " %
                  (step * 1.0 / batch_len, _loss))

        total_loss += _loss
        step += 1
        if step >= batch_len:
            break
    return total_loss / batch_len, _gs, _preds, dec_tgt_batch


def train():
    resout = open(outfile, 'a')

    # ---- Training header: timestamp + key hyperparameters, so appended runs stay distinguishable ----
    print("==================================================", file=resout)
    print("# TRAIN START {} | num_epochs={} batch_size={} lr={} pro={} d_model={}".format(
        _ts(), hp.num_epochs, hp.batch_size, hp.lr, hp.pro, hp.d_model), file=resout)
    print("# ckpt_dir={}".format(hp.ckpt_dir), file=resout)
    print("==================================================", file=resout)
    resout.flush()

    logging.info("# Load model")
    m = Transformer(hp)
    m.loss, m.train_op, m.global_step, m.train_summaries, m.preds = m.train()
    y_hat = m.eval()

    logging.info("# Session")
    saver = tf.compat.v1.train.Saver(max_to_keep=hp.save_epochs)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        ckpt = tf.compat.v1.train.latest_checkpoint(hp.ckpt_dir)
        if ckpt is None:
            logging.info("Initializing from scratch")
            sess.run(tf.compat.v1.global_variables_initializer())
            save_variable_specs(os.path.join(hp.logdir, "specs"))
        else:
            saver.restore(sess, ckpt)

        summary_writer = tf.compat.v1.summary.FileWriter(hp.logdir, sess.graph)
        best_eval_val_f1 = 0
        best_epoch = -1
        epochs_run = 0            # epochs actually run (early stopping may stop before num_epochs)
        no_improve = 0            # consecutive epochs without a new best
        early_stopped = False
        train_start = time.time()
        for i in range(hp.num_epochs):
            epoch_start = time.time()

            train_loss, _global_step, _preds, _tgt = run_epoch(sess, m, 'train_subword', summary_writer, epoch=i)
            _, train_precision, train_recall, train_f1, train_acc = get_hypotheses('train_subword', hp, sess, m, y_hat, m.data.w2id, m.data.id2w, epoch=i)
            print('epoch {}: train precision {}, train recall {}, train f1 {}, train acc {}'.format(i+1, train_precision, train_recall, train_f1, train_acc))


            logging.info("# valiation")

            logging.info("# get hypotheses")

            hypotheses, val_precision, val_recall, val_f1, val_acc = get_hypotheses('eval_subword', hp, sess, m, y_hat, m.data.w2id, m.data.id2w)

            epoch_dur = time.time() - epoch_start
            elapsed = time.time() - train_start
            # Estimated remaining time (linear extrapolation from the average epoch time)
            avg_epoch = elapsed / (i + 1)
            eta = avg_epoch * (hp.num_epochs - (i + 1))

            # ---- Write a full line per epoch: timestamp / elapsed / ETA / train and eval metrics / train_loss ----
            print('epoch {}: eval precision {}, eval recall {}, eval f1 {}, eval acc {}'.format(i+1, val_precision, val_recall, val_f1, val_acc), file=resout)
            print('    [{}] epoch_time={} elapsed={} eta={} | train_loss={:.4f} train_f1={:.4f} train_acc={:.4f} | eval_f1={:.4f} eval_acc={:.4f}'.format(
                _ts(), _fmt_dur(epoch_dur), _fmt_dur(elapsed), _fmt_dur(eta),
                train_loss, train_f1, train_acc, val_f1, val_acc), file=resout)
            resout.flush()
            # print(hypotheses)
            epochs_run = i + 1
            if val_f1 > best_eval_val_f1 + hp.min_delta:
                best_eval_val_f1 = val_f1
                best_epoch = i + 1
                no_improve = 0
                logging.info("# write eval results")
                model_output = "java_E%02dL%.2f" % (i, train_loss)

                logging.info("# save models")
                ckpt_name = os.path.join(hp.ckpt_dir, model_output)
                saver.save(sess, ckpt_name, global_step=_global_step)
                logging.info("after training of {} epochs, {} has been saved.".format(i, ckpt_name))
                # Record provenance of the best checkpoint
                print('    >>> BEST updated: epoch={} eval_f1={:.4f} saved_ckpt={}-{}'.format(
                    i + 1, val_f1, model_output, _global_step), file=resout)
                resout.flush()
            else:
                no_improve += 1
                print('    --- no improve: {}/{} (best eval_f1={:.4f} @ epoch {})'.format(
                    no_improve, hp.patience, best_eval_val_f1, best_epoch), file=resout)
                resout.flush()

            logging.info("# fall back to train mode")
            # sess.run(train_init_op)

            # ---- Early stopping: stop after `patience` epochs without a new best ----
            if hp.patience > 0 and no_improve >= hp.patience:
                early_stopped = True
                print('    >>> EARLY STOP at epoch {} (no improvement for {} epochs; best eval_f1={:.4f} @ epoch {})'.format(
                    i + 1, no_improve, best_eval_val_f1, best_epoch), file=resout)
                resout.flush()
                logging.info("Early stopping triggered at epoch {}.".format(i + 1))
                break

        total_dur = time.time() - train_start
        # ---- End-of-training summary ----
        print("--------------------------------------------------", file=resout)
        print("# TRAIN END {} | total_time={} | epochs_run={}/{} | early_stopped={} | avg_epoch={} | BEST eval_f1={:.4f} @ epoch {}".format(
            _ts(), _fmt_dur(total_dur), epochs_run, hp.num_epochs, early_stopped,
            _fmt_dur(total_dur / max(epochs_run, 1)), best_eval_val_f1, best_epoch), file=resout)
        print("==================================================", file=resout)
        resout.flush()

    summary_writer.close()
    resout.close()

def main(_):
    train()

if __name__ == '__main__':
    tf.compat.v1.app.run()


