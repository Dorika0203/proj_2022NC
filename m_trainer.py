import sys
import time
import os
import argparse
import yaml
import torch
import warnings
import torch.distributed as dist
import torch.multiprocessing as mp
import glob
from torch.utils.data import DataLoader, RandomSampler
from tuneThreshold import *
# from torch.utils.tensorboard import SummaryWriter


from m_network import *
from m_network import *
from m_DataLoader import *
warnings.simplefilter("ignore")

## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description="SpeakerNet")
parser.add_argument('--config', type=str, default=None,
                    help='Config YAML file')

## Data loader
parser.add_argument('--batch_size', type=int, default=1000,
                    help='Batch size, number of speakers per batch')
parser.add_argument('--nDataLoaderThread', type=int,
                    default=5, help='Number of loader threads')
parser.add_argument('--seed', type=int,   default=10,
                    help='Seed for the random number generator')
parser.add_argument('--max_seg_per_spk', type=int,  default=500,
                    help='Maximum number of utterances per speaker per epoch')
parser.add_argument('--multiple_embedding_flag', type=str,  default='B',
                    help='normalization and averaging order of multiple embedder. B: avg-norm-avg, C: avg-avg-norm')
parser.add_argument('--triplet',        dest='triplet', action='store_true',
                    help='data loading in triplet, when using triplet loss.')

## Training details``
parser.add_argument('--test_interval',  type=int,   default=5,
                    help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',      type=int,
                    default=100,    help='Maximum number of epochs')
parser.add_argument('--trainfunc',      type=str,
                    default="MSE",     help='Loss function')

## Optimizer
parser.add_argument('--optimizer',      type=str,
                    default="adam", help='sgd or adam')
parser.add_argument('--scheduler',      type=str,
                    default="steplr", help='Learning rate scheduler')
parser.add_argument('--lr',             type=float,
                    default=0.001,  help='Learning rate')
parser.add_argument("--lr_decay",       type=float, default=0.95,
                    help='Learning rate decay every [test_interval] epochs')
parser.add_argument('--weight_decay',   type=float, default=0,
                    help='Weight decay in the optimizer')

## Loss functions
parser.add_argument('--margin',         type=float, default=0.1,
                    help='Loss margin, only for some loss functions')
parser.add_argument('--scale',          type=float, default=30,
                    help='Loss scale, only for some loss functions')
parser.add_argument('--nPerSpeaker',    type=int,   default=1,
                    help='Number of utterances per speaker per batch, only for metric learning based losses')
parser.add_argument('--nClasses',       type=int,   default=5994,
                    help='Number of speakers in the softmax layer, only for softmax-based losses')
parser.add_argument('--sigma', type=float, default=1,
                    help='Gamma value of RBF kernel, for MMD Loss')

## Load and save
parser.add_argument('--initial_model',  type=str,
                    default="",     help='Initial model weights')
parser.add_argument('--save_path',      type=str,
                    default="exp_emb/temp", help='Path for model and logs')

## Training and test data
parser.add_argument('--train_list',     type=str,
                    default="data/train_list.txt",  help='Train list, 1 emb file per line')
parser.add_argument('--valid_list',     type=str,   default="data/valid_list.txt",
                    help='Validation list, same as train list with different dataset')
parser.add_argument('--test_list',      type=str,
                    default="data/test_list.txt",   help='Test list, 2 emb files per line')
parser.add_argument('--test_list_libri',      type=str,
                    default="data/valid2_test_list.txt",   help='Libri test list, 2 emb files per line')


parser.add_argument('--train_path',     type=str,
                    default="/home/doyeolkim/vox_emb/train/", help='Absolute path to the train set')
parser.add_argument('--valid_path',     type=str,
                    default="/home/doyeolkim/vox_emb/test/", help='Absolute path to the valid set')
parser.add_argument('--test_path',      type=str,
                    default="/home/doyeolkim/vox_emb/test/", help='Absolute path to the test set')
parser.add_argument('--test_path_libri',      type=str,
                    default="/home/doyeolkim/libri_emb/valid2/", help='Absolute path to the libri test set')


## Model definition
parser.add_argument('--model',          type=str,
                    default="m_LinearNet", help='Name of model definition')
parser.add_argument('--nOut',           type=int,   default=512,
                    help='Embedding size in the last FC layer')
parser.add_argument('--activation',     type=str,   default="ReLU",
                    help='activation function between FC layers')

## For test only
parser.add_argument('--eval',           dest='eval',
                    action='store_true', help='Eval only')
parser.add_argument('--check',           dest='check',
                    action='store_true', help='Check EER of original method')
parser.add_argument('--generate',        dest='generate',
                    action='store_true', help='Genearting Embedding Flag')

## Distributed and mixed precision training
parser.add_argument('--port',           type=str,   default="8888",
                    help='Port for distributed training, input as text')
parser.add_argument('--distributed',    dest='distributed',
                    action='store_true', help='Enable distributed training')


# 모델 이용 임베딩 생성
parser.add_argument('--gen_list', type=str,
                    default='data/testset_normal_list', help='Generating List, normal type')
parser.add_argument('--gen_path', type=str, default='/home/doyeolkim/vox_emb/test/',
                    help='Absolute path to the generating set')
parser.add_argument('--gen_target_dir', type=str, default='emb',
                    help='Generating embedding target directory')


args = parser.parse_args()

## Parse YAML


def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
           return opt.type
    raise ValueError


if args.config is not None:

    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__:
            typ = find_option_type(k, parser)
            args.__dict__[k] = typ(v)
        else:
            sys.stderr.write(
                "Ignored unknown parameter {} in yaml.\n".format(k))

    # set args.save to config file name
    args.save_path = 'exp_emb/' + args.config.strip().split('/')[-1][:-5]


def main_worker(gpu, ngpus_per_node, args):

    args.gpu = gpu

    ## Load models
    s = EmbedNet(**vars(args))

    if args.distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.port

        dist.init_process_group(
            backend='nccl', world_size=ngpus_per_node, rank=args.gpu)

        torch.cuda.set_device(args.gpu)
        s.cuda(args.gpu)

        # s = torch.nn.parallel.DistributedDataParallel(s, device_ids=[args.gpu], find_unused_parameters=False)
        s = torch.nn.parallel.DistributedDataParallel(
            s, device_ids=[args.gpu], find_unused_parameters=True)

        print('Loaded the model on GPU {:d}'.format(args.gpu))

    else:
        s = WrappedModel(model=s).cuda(args.gpu)

    it = 1

    if args.gpu == 0:
        ## Write args to scorefile
        scorefile = open(args.result_save_path+"/scores.txt", "a+")

    ## Initialise trainer and data loader
    if args.trainfunc[0:2] == 'DA':
        args.batch_size = 1
        train_dataset = MyDistributionDataset(
            args.train_list, args.train_path, **vars(args))
    else:
        train_dataset = MyDataset(
            args.train_list, args.train_path, **vars(args))

    if args.nPerSpeaker != 1:
        train_sampler = train_dataset_sampler(
            data_source=train_dataset, **vars(args))
    else:
        if args.distributed:
            train_sampler = DistributedSampler(train_dataset)
        else:
            train_sampler = RandomSampler(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.nDataLoaderThread,
                              sampler=train_sampler, pin_memory=False, worker_init_fn=worker_init_fn, drop_last=True)

    trainer = ModelTrainer(s, **vars(args))

    ## Load model weights
    modelfiles = glob.glob('%s/model0*.model' % args.model_save_path)
    modelfiles.sort()

    if(args.initial_model != ""):
        trainer.loadParameters(args.initial_model)
        print("Model {} loaded!".format(args.initial_model))
    elif len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1])
        print("Model {} loaded from previous state!".format(modelfiles[-1]))
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1

    for ii in range(1, it):
        trainer.__scheduler__.step()
        
        
        
    # 임베딩 생성 (Generate list 이용, normal list 형태, 화자 정보는 무시)
    if args.generate == True:
        
        # if train function is DA, change it to MSE
        args.trainfunc = 'MSE' if args.trainfunc == 'DA' else args.trainfunc
        
        s = EmbedNet(**vars(args))
        s = WrappedModel(model=s).cuda(args.gpu)
        s.eval()
        
        
        if args.gpu == 0:
            generate_dataset = MyDataset(args.gen_list, args.gen_path, **vars(args))
            generate_loader = DataLoader(generate_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
            emb_save_path = os.path.join(args.save_path, args.gen_target_dir)
            
            with open(args.gen_list) as generate_file:
                lines = generate_file.readlines()
            
            for idx, (sing_emb, _) in enumerate(generate_loader):
                
                tokens = lines[idx].split()[-1].split('/')
                spk = tokens[0]
                cat = tokens[1]
                filename = tokens[2]
                emb = s(sing_emb).detach().cpu()[0]
                
                emb_save_path_final = os.path.join(emb_save_path, spk)
                emb_save_path_final = os.path.join(emb_save_path_final, cat)
                os.makedirs(emb_save_path_final, exist_ok=True)
                
                numpy.save(os.path.join(emb_save_path_final, filename),numpy.array(emb))
                print("\r{}/{}".format(idx+1, len(generate_loader)), end='')
        return





    ## Get Original EER
    if args.check == True:
        sc, lab, tr = trainer.get_original_result(**vars(args))
        sc2, lab2, tr2 = trainer.compareProcessedSingleEmbs(**vars(args))
        if args.gpu == 0:
            result = tuneThresholdfromScore(sc, lab, [1, 0.1])
            result2 = tuneThresholdfromScore(sc2, lab2, [1, 0.1])
            print('\n', time.strftime("%Y-%m-%d %H:%M:%S"),
                  "[testset original EER] {:2.4f}, [testset processed EER] {:2.4f}".format(result[1], result2[1]))

        args2 = argparse.Namespace(**vars(args))
        args2.test_list = args2.test_list_libri
        args2.test_path = args2.test_path_libri
        sc, lab, tr = trainer.get_original_result(**vars(args2))
        sc2, lab2, tr2 = trainer.compareProcessedSingleEmbs(**vars(args2))
        if args.gpu == 0:
            result = tuneThresholdfromScore(sc, lab, [1, 0.1])
            result2 = tuneThresholdfromScore(sc2, lab2, [1, 0.1])
            print('\n', time.strftime("%Y-%m-%d %H:%M:%S"),
                  "[libriset original EER] {:2.4f}, [libriset processed EER] {:2.4f}".format(result[1], result2[1]))

        return

    ## Core training script
    for it in range(it, args.max_epoch+1):

        if args.distributed:
            train_sampler.set_epoch(it)

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        loss = trainer.train_network(train_loader, verbose=(args.gpu == 0))

        if args.gpu == 0:
            print('\n', time.strftime("%Y-%m-%d %H:%M:%S"),
                  "Epoch {:d}, TLOSS {:f}, LR {:f}".format(it, loss, max(clr)))
            scorefile.write(
                "Epoch {:d}, TLOSS {:f}, LR {:f} \n".format(it, loss, max(clr)))
            scorefile.flush()

        '''
        Validation 파트
        '''
        if it % args.test_interval == 0:

            mean_loss, mean_prec = trainer.validationLoss(**vars(args))
            sc, lab, _ = trainer.compareProcessedSingleEmbs(**vars(args))

            args2 = argparse.Namespace(**vars(args))
            args2.test_list = args2.test_list_libri
            args2.test_path = args2.test_path_libri
            sc2, lab2, _ = trainer.compareProcessedSingleEmbs(**vars(args2))

            if args.gpu == 0:
                result = tuneThresholdfromScore(sc, lab, [1, 0.1])
                result2 = tuneThresholdfromScore(sc2, lab2, [1, 0.1])
                print('\n', ' Epoch {:d}, VLoss {:2.6f}, VEER {:2.4f}, VEER_LIBRI {:2.4f}, Vacc {:2.4f}'.format(
                    it, mean_loss, result[1], result2[1], mean_prec))
                scorefile.write("--Val-- Epoch {:d}, VLoss {:2.6f}, VEER {:2.4f}, VEER_LIBRI {:2.4f}, Vacc {:2.4f}\n".format(
                    it, mean_loss, result[1], result2[1], mean_prec))
                trainer.saveParameters(
                    args.model_save_path+"/model%09d.model" % it)
                scorefile.flush()



## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====


def main():
    # Generate Embedding Vector
    args.model_save_path = args.save_path+"/model"
    args.result_save_path = args.save_path+"/result"
    args.feat_save_path = ""

    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)

    n_gpus = torch.cuda.device_count()

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    print('Save path:', args.save_path)

    if args.distributed:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)


if __name__ == '__main__':
    main()
