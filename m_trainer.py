import sys, time, os, argparse
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

parser = argparse.ArgumentParser(description = "SpeakerNet")
parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file')

## Data loader
parser.add_argument('--batch_size',     type=int,   default=1000,    help='Batch size, number of speakers per batch')
parser.add_argument('--nDataLoaderThread', type=int, default=5,     help='Number of loader threads')
parser.add_argument('--seed',           type=int,   default=10,     help='Seed for the random number generator')
parser.add_argument('--max_seg_per_spk', type=int,  default=500,    help='Maximum number of utterances per speaker per epoch')

## Training details``
parser.add_argument('--test_interval',  type=int,   default=5,     help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',      type=int,   default=100,    help='Maximum number of epochs')
parser.add_argument('--trainfunc',      type=str,   default="MSE",     help='Loss function')

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam')
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler')
parser.add_argument('--lr',             type=float, default=0.001,  help='Learning rate')
parser.add_argument("--lr_decay",       type=float, default=0.95,   help='Learning rate decay every [test_interval] epochs')
parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer')

## Loss functions
parser.add_argument('--margin',         type=float, default=0.1,    help='Loss margin, only for some loss functions')
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions')
parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Number of utterances per speaker per batch, only for metric learning based losses')
parser.add_argument('--nClasses',       type=int,   default=5994,   help='Number of speakers in the softmax layer, only for softmax-based losses')

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights')
parser.add_argument('--save_path',      type=str,   default="exp_emb/temp", help='Path for model and logs')

## Training and test data
parser.add_argument('--train_list',     type=str,   default="data/train_list.txt",  help='Train list, 1 emb file per line')
parser.add_argument('--valid_list',     type=str,   default="data/valid_list.txt",  help='Validation list, same as train list with different dataset')
parser.add_argument('--test_list',      type=str,   default="data/test_list.txt",   help='Test list, 2 emb files per line')
parser.add_argument('--train_path',     type=str,   default="/home/doyeolkim/vox_emb/train/", help='Absolute path to the train set')
parser.add_argument('--valid_path',     type=str,   default="/home/doyeolkim/vox_emb/test/", help='Absolute path to the valid set')
parser.add_argument('--test_path',      type=str,   default="/home/doyeolkim/vox_emb/test/", help='Absolute path to the test set')

## Model definition
parser.add_argument('--model',          type=str,   default="m_LinearNet", help='Name of model definition')
parser.add_argument('--nOut',           type=int,   default=512,           help='Embedding size in the last FC layer')
parser.add_argument('--activation',     type=str,   default="ReLU",        help='activation function between FC layers')

## For test only
parser.add_argument('--eval',           dest='eval', action='store_true', help='Eval only')
parser.add_argument('--check',           dest='check', action='store_true', help='Check EER of original method')

## Distributed and mixed precision training
parser.add_argument('--port',           type=str,   default="8888", help='Port for distributed training, input as text')
parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')

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
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))
    
    
    # set args.save to config file name
    args.save_path = 'exp_emb/' + args.config.strip().split('/')[-1][:-5]
    
    
def main_worker(gpu, ngpus_per_node, args):
    
    args.gpu = gpu

    ## Load models
    s = EmbedNet(**vars(args))

    if args.distributed:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']=args.port

        dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)

        torch.cuda.set_device(args.gpu)
        s.cuda(args.gpu)

        # s = torch.nn.parallel.DistributedDataParallel(s, device_ids=[args.gpu], find_unused_parameters=False)
        s = torch.nn.parallel.DistributedDataParallel(s, device_ids=[args.gpu], find_unused_parameters=True)

        print('Loaded the model on GPU {:d}'.format(args.gpu))

    else:
        s = WrappedModel(model=s).cuda(args.gpu)

    it = 1

    if args.gpu == 0:
        ## Write args to scorefile
        scorefile   = open(args.result_save_path+"/scores.txt", "a+")

    ## Initialise trainer and data loader
    train_dataset = MyDataset(args.train_list, args.train_path, **vars(args))
    
    if args.nPerSpeaker != 1:
        train_sampler = train_dataset_sampler(data_source=train_dataset, **vars(args))
    else:
        if args.distributed:
            train_sampler = DistributedSampler(train_dataset)
        else:
            train_sampler = RandomSampler(train_dataset)
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.nDataLoaderThread,
        sampler=train_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    trainer = ModelTrainer(s, **vars(args))

    ## Load model weights
    modelfiles = glob.glob('%s/model0*.model'%args.model_save_path)
    modelfiles.sort()

    if(args.initial_model != ""):
        trainer.loadParameters(args.initial_model)
        print("Model {} loaded!".format(args.initial_model))
    elif len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1])
        print("Model {} loaded from previous state!".format(modelfiles[-1]))
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1

    for ii in range(1,it):
        trainer.__scheduler__.step()



    ## Evaluation code - must run on single GPU
    if args.eval == True:
        pytorch_total_params = sum(p.numel() for p in s.module.__S__.parameters())
        print('Total parameters: ',pytorch_total_params)
        print('Test list',args.test_list)
        sc, lab, _ = trainer.compareProcessedSingleEmbs(**vars(args))
        if args.gpu == 0:
            result = tuneThresholdfromScore(sc, lab, [1, 0.1])
            print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Test EER {:2.4f}".format(result[1]))
        return
    
    
    ## Get Original EER
    if args.check == True:
        sc, lab, _ = trainer.get_original_result(**vars(args))
        if args.gpu == 0:
            result = tuneThresholdfromScore(sc, lab, [1, 0.1])
            print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Test EER {:2.4f}".format(result[1]))
        return

    ## Core training script
    
    for it in range(it,args.max_epoch+1):
        
        if args.distributed:
            train_sampler.set_epoch(it)

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        loss = trainer.train_network(train_loader, verbose=(args.gpu == 0))

        if args.gpu == 0:
            print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, TLOSS {:f}, LR {:f}".format(it, loss, max(clr)))
            scorefile.write("Epoch {:d}, TLOSS {:f}, LR {:f} \n".format(it, loss, max(clr)))
            scorefile.flush()
            
            
        '''
        Validation 파트
        '''
        if it % args.test_interval == 0:

            mean_loss, mean_prec = trainer.validationLoss(**vars(args))
            sc, lab, _ = trainer.compareProcessedSingleEmbs(**vars(args))
            
            if args.gpu == 0:
                result = tuneThresholdfromScore(sc, lab, [1, 0.1])
                print('\n',' Epoch {:d}, VLoss {:2.6f}, VEER {:2.4f}, Vacc {:2.4f}'.format(it, mean_loss, result[1], mean_prec))
                scorefile.write("--Val-- Epoch {:d}, VLoss {:2.6f}, VEER {:2.4f}\n".format(it, mean_loss, result[1]))
                trainer.saveParameters(args.model_save_path+"/model%09d.model"%it)
                scorefile.flush()
                
                
    # Save Result
    if args.gpu == 0:
        scorefile.close()
        generate_graph(**vars(args))



## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====


def main():
    # Generate Embedding Vector
    args.model_save_path     = args.save_path+"/model"
    args.result_save_path    = args.save_path+"/result"
    args.feat_save_path      = ""

    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)

    n_gpus = torch.cuda.device_count()

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    print('Save path:',args.save_path)

    if args.distributed:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)

if __name__ == '__main__':
    main()