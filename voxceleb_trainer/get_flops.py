
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from gluoncv.torch.model_zoo import get_model
from gluoncv.torch.engine.config import get_cfg_defaults

from thop import profile, clever_format
##
from SpeakerNet import *
import yaml
import sys
##


def find_option_type(key, parser):
      for opt in parser._get_optional_actions():
         if ('--' + key) in opt.option_strings:
            return opt.type
      raise ValueError

##
class WrappedModel(nn.Module):
    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU

    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None):
        return self.module(x, label)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute FLOPs of a model.')
    parser.add_argument('--config', type=str, help='path to config file.')
    parser.add_argument('--num-frames', type=int, default=32, help='temporal clip length.')
    parser.add_argument('--input-size', type=int, default=224,
                        help='size of the input image size. default is 224')
    
    
    ####

    parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training')
    parser.add_argument('--eval_frames',    type=int,   default=300,    help='Input length to the network for testing 0 uses the whole files')
    parser.add_argument('--batch_size',     type=int,   default=200,    help='Batch size, number of speakers per batch')
    parser.add_argument('--max_seg_per_spk', type=int,  default=500,    help='Maximum number of utterances per speaker per epoch')
    parser.add_argument('--nDataLoaderThread', type=int, default=5,     help='Number of loader threads')
    parser.add_argument('--augment',        type=bool,  default=False,  help='Augment input')
    parser.add_argument('--seed',           type=int,   default=10,     help='Seed for the random number generator')
    parser.add_argument('--test_interval',  type=int,   default=10,     help='Test and save every [test_interval] epochs')
    parser.add_argument('--max_epoch',      type=int,   default=500,    help='Maximum number of epochs')
    parser.add_argument('--trainfunc',      type=str,   default="",     help='Loss function')

    ## Optimizer
    parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam')
    parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler')
    parser.add_argument('--lr',             type=float, default=0.001,  help='Learning rate')
    parser.add_argument("--lr_decay",       type=float, default=0.95,   help='Learning rate decay every [test_interval] epochs')
    parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer')

    ## Loss functions
    parser.add_argument("--hard_prob",      type=float, default=0.5,    help='Hard negative mining probability, otherwise random, only for some loss functions')
    parser.add_argument("--hard_rank",      type=int,   default=10,     help='Hard negative mining rank in the batch, only for some loss functions')
    parser.add_argument('--margin',         type=float, default=0.1,    help='Loss margin, only for some loss functions')
    parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions')
    parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Number of utterances per speaker per batch, only for metric learning based losses')
    parser.add_argument('--nClasses',       type=int,   default=5994,   help='Number of speakers in the softmax layer, only for softmax-based losses')

    ## Evaluation parameters
    parser.add_argument('--dcf_p_target',   type=float, default=0.05,   help='A priori probability of the specified target speaker')
    parser.add_argument('--dcf_c_miss',     type=float, default=1,      help='Cost of a missed detection')
    parser.add_argument('--dcf_c_fa',       type=float, default=1,      help='Cost of a spurious detection')

    ## Load and save
    parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights')
    parser.add_argument('--save_path',      type=str,   default="exps/exp1", help='Path for model and logs')

    ## Training and test data
    parser.add_argument('--train_list',     type=str,   default="data/train_list.txt",  help='Train list')
    parser.add_argument('--test_list',      type=str,   default="data/test_list.txt",   help='Evaluation list')
    parser.add_argument('--train_path',     type=str,   default="data/voxceleb2", help='Absolute path to the train set')
    parser.add_argument('--test_path',      type=str,   default="data/voxceleb1", help='Absolute path to the test set')
    parser.add_argument('--musan_path',     type=str,   default="data/musan_split", help='Absolute path to the test set')
    parser.add_argument('--rir_path',       type=str,   default="data/RIRS_NOISES/simulated_rirs", help='Absolute path to the test set')

    ## Model definition
    parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks')
    parser.add_argument('--log_input',      type=bool,  default=False,  help='Log input features')
    parser.add_argument('--model',          type=str,   default="",     help='Name of model definition')
    parser.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder')
    parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer')
    parser.add_argument('--sinc_stride',    type=int,   default=10,    help='Stride size of the first analytic filterbank layer of RawNet3')

    ## For test only
    parser.add_argument('--eval',           dest='eval', action='store_true', help='Eval only')

    ## Distributed and mixed precision training
    parser.add_argument('--port',           type=str,   default="8888", help='Port for distributed training, input as text')
    parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')
    parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')  
    parser.add_argument('--usecpu', type=bool, default = False, help='use cpu deactivate cuda')





  #####
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config)
    ##
    

    if args.config is not None:
      with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__:
            typ = find_option_type(k, parser)
            args.__dict__[k] = typ(v)
        else:
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))
    
    args.gpu=0
    model = SpeakerNet(**vars(args))
    model = WrappedModel(model)
    
    ##
    #trainer = ModelTrainer(model, **vars(args)) 
   # model = trainer.loadParameters(args.initial_model)
    ###


    input_tensor = torch.autograd.Variable(torch.rand(1, 3, args.num_frames, args.input_size, args.input_size))
    #input_tensor = torch.autograd.Variable(torch.rand(1, 3, args.input_size, args.input_size))
    
    import pdb;pdb.set_trace()
    
    print(input_tensor.shape)
    print(model)
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    
    #print(macs.size())
    #print(params.size())
    print("FLOPs: ", macs, "; #params: ", params)











