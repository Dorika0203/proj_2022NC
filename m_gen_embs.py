import sys, time, os, argparse
import yaml
import numpy as np
import torch
import warnings
from tuneThreshold import *
from SpeakerNet import *
from DatasetLoader import *
import torch.distributed as dist
import torch.multiprocessing as mp
warnings.simplefilter("ignore")
import shutil



'''
DB_PATH 디렉토리 내의 모든 wav 파일들에 대해
똑같은 디렉토리 구조로 EMBED_DIR 내에 임베딩 벡터들을 생성한다.
임베딩 벡터 생성은 Clova ResNetSE34V2를 이용.
'''

# DB_PATH = "/SGV/speechdb/OpenDevSet/VoxCeleb1/test_wav/"
# EMBED_DIR = '/home/doyeolkim/vox_emb/test/'

# DB_PATH = "/SGV/speechdb/OpenDevSet/VoxCeleb2/dev/aac/"
# EMBED_DIR = '/home/doyeolkim/vox_emb/train/'

# DB_PATH = "/home/doyeolkim/libri/train-clean-100/"
# DB_PATH = "/home/doyeolkim/libri/train-clean-360/"
DB_PATH = "/home/doyeolkim/libri/dev-clean/"
EMBED_DIR = '/home/doyeolkim/libri_emb/valid2/'

## ===== ===== ===== ===== ===== ===== ===== =====
## Default parser args
## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = "SpeakerNet")
parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file')
## Data loader
parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training')
parser.add_argument('--eval_frames',    type=int,   default=400,    help='Input length to the network for testing 0 uses the whole files')
parser.add_argument('--batch_size',     type=int,   default=200,    help='Batch size, number of speakers per batch')
parser.add_argument('--max_seg_per_spk', type=int,  default=500,    help='Maximum number of utterances per speaker per epoch')
parser.add_argument('--nDataLoaderThread', type=int, default=5,     help='Number of loader threads')
parser.add_argument('--augment',        type=bool,  default=False,  help='Augment input')
parser.add_argument('--seed',           type=int,   default=10,     help='Seed for the random number generator')
## Training details
parser.add_argument('--test_interval',  type=int,   default=10,     help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',      type=int,   default=500,    help='Maximum number of epochs')
parser.add_argument('--trainfunc',      type=str,   default="softmaxproto",     help='Loss function')
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
parser.add_argument('--initial_model',  type=str,   default="baseline_v2_ap.model",     help='Initial model weights')
parser.add_argument('--save_path',      type=str,   default="exps/test", help='Path for model and logs')
## Training and test data
parser.add_argument('--train_list',     type=str,   default="data/train_list.txt",  help='Train list')
parser.add_argument('--test_list',      type=str,   default="data/test_list.txt",   help='Evaluation list')
parser.add_argument('--train_path',     type=str,   default="data/voxceleb2", help='Absolute path to the train set')
parser.add_argument('--test_path',      type=str,   default="data/voxceleb1", help='Absolute path to the test set')
parser.add_argument('--musan_path',     type=str,   default="data/musan_split", help='Absolute path to the test set')
parser.add_argument('--rir_path',       type=str,   default="data/RIRS_NOISES/simulated_rirs", help='Absolute path to the test set')
## Model definition
parser.add_argument('--n_mels',         type=int,   default=64,     help='Number of mel filterbanks')
parser.add_argument('--log_input',      type=bool,  default=True,  help='Log input features')
parser.add_argument('--model',          type=str,   default="ResNetSE34V2",     help='Name of model definition')
parser.add_argument('--encoder_type',   type=str,   default="ASP",  help='Type of encoder')
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer')
parser.add_argument('--sinc_stride',    type=int,   default=10,    help='Stride size of the first analytic filterbank layer of RawNet3')
## For test only
parser.add_argument('--eval',           dest='eval', action='store_true', help='Eval only')
## Distributed and mixed precision training
parser.add_argument('--port',           type=str,   default="8888", help='Port for distributed training, input as text')
parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')
parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')
args = parser.parse_args(args=[])

## Parse YAML
def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
           return opt.type
    raise ValueError

## ===== ===== ===== ===== ===== ===== ===== =====
## config 파일 경로로 설정할 거면 여기
args.config = None
## ===== ===== ===== ===== ===== ===== ===== =====

if args.config is not None:
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__:
            typ = find_option_type(k, parser)
            args.__dict__[k] = typ(v)
        else:
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))


def get_wav_list(target_root):
    '''
    target_root 내 wav file들 경로 포함 이름 반환
    '''
    ret_list = []
    
    for root, directories, filenames in os.walk(target_root):
        for filename in filenames:
            if str.endswith(filename, '.wav'):
                file_str = root[len(target_root):] + '/' + filename
                ret_list.append(file_str)
    
    return ret_list

def ignore_files(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f)) or os.path.islink(os.path.join(dir, f))]

class MyDatasetLoader(Dataset):
    def __init__(self, test_list, test_path, eval_frames, num_eval, **kwargs):
        self.max_frames = eval_frames
        self.num_eval   = num_eval
        self.test_path  = test_path
        self.test_list  = test_list

    def __getitem__(self, index):
        audio = loadWAV(os.path.join(self.test_path , self.test_list[index]), self.max_frames, evalmode=True, num_eval=self.num_eval)
        return torch.FloatTensor(audio), self.test_list[index]

    def __len__(self):
        return len(self.test_list)
    
    
    
def main_worker(gpu, ngpus_per_node, args):
    
    if os.path.exists(EMBED_DIR):
        shutil.rmtree(EMBED_DIR)
    print('..... Tree removal Done')
    
    # directory structure copy
    shutil.copytree(DB_PATH, EMBED_DIR, ignore=ignore_files, dirs_exist_ok=True)
    print('..... Tree Copy Done')
    
    wav_list = get_wav_list(DB_PATH)
    
    args.gpu = gpu
    
    s = SpeakerNet(**vars(args))
    s.eval()
    
    # args.distributed=True

    if args.distributed:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']=args.port
        dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)
        torch.cuda.set_device(args.gpu)
        s.cuda(args.gpu)
        s = torch.nn.parallel.DistributedDataParallel(s, device_ids=[args.gpu], find_unused_parameters=True)
        print('Loaded the model on GPU {:d}'.format(args.gpu))
    else:
        s = WrappedModel(s).cuda(args.gpu)
        
    # pytorch_total_params = sum(p.numel() for p in s.module.__S__.parameters())
    # print('Total parameters: ',pytorch_total_params)
    
    trainer = ModelTrainer(s, **vars(args))
    if args.initial_model != "":
        trainer.loadParameters(args.initial_model)
        print("Model {} loaded!".format(args.initial_model))
        
    args.test_list = wav_list
    args.test_path = DB_PATH
    
    embedding_dataset = MyDatasetLoader(num_eval=10, **vars(args))
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(embedding_dataset, shuffle=False)
    else:
        sampler = None
    embedding_dataloader = torch.utils.data.DataLoader(embedding_dataset, batch_size=1, shuffle=False, num_workers=args.nDataLoaderThread, drop_last=False, sampler=sampler)
    
    out_list = []
    
    for idx, data in enumerate(embedding_dataloader):
        with torch.no_grad():
            out = s.forward(data[0][0]).detach().cpu().numpy()
        out_list.append(out)
        print('\r...processing ({}/{})'.format(idx+1, len(wav_list)), end="")
    
    print()    
    for idx, wav_name in enumerate(wav_list):
        npy_name = EMBED_DIR + wav_name[:-3] + 'npy'
        np.save(npy_name, out_list[idx])
        print('\r...saving ({}/{})'.format(idx+1, len(wav_list)), end="")
        
    print()
    return




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