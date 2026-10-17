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
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
save_hparams(hp, hp.logdir)
os.environ['CUDA_VISIBLE_DEVICES'] = hp.gpu
outfile = hp.res_log


def test():
    resout = open(outfile, 'a')
    
    logging.info("# Load model")
    m = Transformer(hp)
    y_hat = m.eval()

    logging.info("# Session")
    saver = tf.compat.v1.train.Saver(max_to_keep=hp.save_epochs)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        ckpt = tf.compat.v1.train.latest_checkpoint(hp.ckpt_dir)
        saver.restore(sess, ckpt)

        summary_writer = tf.compat.v1.summary.FileWriter(hp.logdir, sess.graph)

        # test
        test_start = time.time()
        test_hypotheses, test_precision, test_recall, test_f1, test_acc = get_hypotheses('test_subword', hp, sess, m, y_hat, m.data.w2id, m.data.id2w)
        test_dur = time.time() - test_start
        print('test precision {}, test recall {}, test f1 {}, test acc {}'.format(test_precision, test_recall, test_f1, test_acc), file=resout)
        # ---- Test provenance: timestamp / checkpoint / test directory / decoding time ----
        print('    [{}] ckpt={} testdir={} decode_time={:.1f}s'.format(
            _ts(), ckpt, hp.testdir, test_dur), file=resout)
        resout.flush()
    
    summary_writer.close()
    resout.close()

def main(_):
    test()

if __name__ == '__main__':
    tf.compat.v1.app.run()


