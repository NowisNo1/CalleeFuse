import argparse

class Hparams:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', help='gpu id')
    

    # dataset name
    dataset_name = "java-large/"
    # training data path
    train_dir = "data_processed/"+dataset_name
    # validation data path
    eval_dir = "data_processed/"+dataset_name
    # test data path
    test_dir = "data_processed/"+dataset_name
    # base directory for model outputs
    dir_prefix = 'saved/'+dataset_name
    # per-model output directory
    model_dir = dir_prefix + 'train_with_pro_with_doc/'
    # Number of epochs (early-stopping cap, not a fixed count)
    # Note: originally 25, but mask_with_ctx / original still had rising validation F1 at 25 epochs (not converged).
    # With early stopping this is only a safety cap: converged experiments stop early at the plateau.
    # Slow-converging experiments may plateau after epoch 50, so the cap is 100 to let early stopping decide.
    num_epochs = 100
    # training batch size
    batch_size = 64
    # learning rate
    lr = 0.0001
    # whether to use project-specific information
    use_project_info = True
    # whether to use docstring information
    use_docstring = True

    '''
    The following are configurable; other parameters should not be changed.
    '''
    # ======================== configurable options ========================

    # ========== training hyperparameters ==========
    # training batch size
    parser.add_argument('--batch_size', default=batch_size, type=int)
    # evaluation batch size
    parser.add_argument('--eval_batch_size', default=128, type=int)
    # log directory
    parser.add_argument('--logdir', default=model_dir+'log/', help="log directory")
    # number of epochs (early-stopping cap; training stops early after convergence)
    parser.add_argument('--num_epochs', default=num_epochs, type=int)
    # early-stopping patience: stop after this many epochs without a new best eval_f1 (<=0 disables)
    parser.add_argument('--patience', default=5, type=int,
                        help="early stopping patience (epochs without eval_f1 improvement); <=0 disables")
    # minimum improvement: an eval_f1 gain must exceed this to count as a new best
    # Set to 0.001 so that fluctuations of about +0.0001 do not count as a new best and experiments that already plateaued
    # do not keep resetting the patience counter and run to the cap (false non-convergence).
    parser.add_argument('--min_delta', default=0.001, type=float,
                        help="minimum eval_f1 improvement to count as a new best")
    # checkpoint save interval
    parser.add_argument('--save_epochs', default=3, type=int)
    # evaluation directory
    parser.add_argument('--evaldir', default=eval_dir, help="evaluation dir")

    # whether to use project-specific information
    parser.add_argument('--pro', default=use_project_info,
                        type=lambda x: x.lower() in ('true', '1', 'yes'),
                        help="whether to use project-specific info")
    # whether to use docstring information
    parser.add_argument('--use_docstring', default=use_docstring,
                        type=lambda x: x.lower() in ('true', '1', 'yes'),
                        help="whether to use docstring information")
    
    # ========== test hyperparameters ==========
    # test batch size
    parser.add_argument('--test_batch_size', default=128, type=int)
    # test directory
    parser.add_argument('--testdir', default=test_dir, help="test result dir")
    # result directory
    parser.add_argument('--res_log', default=model_dir+'res.txt', help="result dir")

    # ========== checkpoint paths ==========
    parser.add_argument('--ckpt_dir', default=model_dir+'ckpt/', help="checkpoint file path")

    # ======================== configurable options ========================

    # ========== data paths ==========
    # vocabulary file path
    parser.add_argument('--sub_word_vocab_file', default='data_processing/sub_token_w2id.txt',
                        help="vocabulary file path")
    parser.add_argument('--doc_vocab_file', default='data_processing/doc_w2id.txt',
                        help="vocabulary file path")
    # training data path
    parser.add_argument('--data_path', default=train_dir,
                        help="data path")

    '''
    The following should not be modified.
    '''
    # ====================== do not modify ======================
    # learning rate
    parser.add_argument('--lr', default=lr, type=float, help="learning rate")
    # warmup steps
    parser.add_argument('--warmup_steps', default=4000, type=int)
    # ========== context sizes ==========
    # body context size
    parser.add_argument('--body_context_size', default=55, type=int,
                        help="body_context_size")
    # project context size
    parser.add_argument('--project_context_size', default=60, type=int,
                        help="project_context_size")
    # docstring context size
    parser.add_argument('--doc_context_size', default=10, type=int,
                        help="doc_context_size")
    # target-name context size
    parser.add_argument('--tgt_name_size', default=5, type=int,
                        help="tgt_name_size")

    # ========== model hyperparameters ==========
    # model dimension
    parser.add_argument('--d_model', default=512, type=int,
                        help="hidden dimension of encoder/decoder")
    # feed-forward dimension
    parser.add_argument('--d_ff', default=2048, type=int,
                        help="hidden dimension of feedforward layer")
    # number of encoder/decoder blocks
    parser.add_argument('--num_blocks', default=6, type=int,
                        help="number of encoder/decoder blocks")
    # number of attention heads
    parser.add_argument('--num_heads', default=8, type=int,
                        help="number of attention heads")
    # dropout rate
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    # label smoothing rate
    parser.add_argument('--smoothing', default=0.1, type=float,
                        help="label smoothing rate")
