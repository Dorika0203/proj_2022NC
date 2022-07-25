import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy, sys, random
import time, itertools, importlib

from m_DataLoader import MyDataset
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler


class WrappedModel(nn.Module):

    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU

    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None):
        return self.module(x, label)


class ModelTrainer(object):
    def __init__(self, speaker_model, optimizer, scheduler, gpu, **kwargs):

        self.__model__ = speaker_model
        
        Optimizer = importlib.import_module("optimizer." + optimizer).__getattribute__("Optimizer")
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)
        
        Scheduler = importlib.import_module("scheduler." + scheduler).__getattribute__("Scheduler")
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)
        
        self.scaler = GradScaler()
        self.gpu = gpu

        assert self.lr_step in ["epoch", "iteration"]

    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader, verbose):

        self.__model__.train()

        stepsize = loader.batch_size

        counter = 0
        index = 0
        loss = 0

        tstart = time.time()
        # breakpoint()

        for data, data_label in loader:
        # for _, (data, data_label) in enumerate(tqdm(loader)):
    
            # data = data.transpose(1, 0)
            self.__model__.zero_grad()
            nloss = self.__model__(data, data_label)
            nloss.backward()
            self.__optimizer__.step()
                
            loss += nloss.detach().cpu().item()
            counter += 1
            index += stepsize

            telapsed = time.time() - tstart
            tstart = time.time()

            if verbose:
                sys.stdout.write("\rProcessing {:d} of {:d}:".format(index, loader.__len__() * loader.batch_size))
                sys.stdout.write(" Loss {:f} - {:.2f} Hz ".format(loss / counter, stepsize / telapsed))
                sys.stdout.flush()

            if self.lr_step == "iteration":
                self.__scheduler__.step()

        if self.lr_step == "epoch":
            self.__scheduler__.step()

        return (loss / counter)














    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList(self, test_list, test_path, nDataLoaderThread, distributed, print_interval=1, num_eval=10, **kwargs):

        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        self.__model__.eval()

        ## Define test data loader
        test_dataset = MyDataset(test_list, test_path, **kwargs)

        # test_sampler = TrainSampler(train_dataset, **vars(args))
        if distributed:
            test_sampler = DistributedSampler(test_dataset)
        else:
            test_sampler = SequentialSampler(test_dataset)
            
            
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=nDataLoaderThread,
            drop_last=False,
            sampler=test_sampler    
        )
        
        loss = []
        L = 0
        
        for idx, (data, label) in enumerate(test_loader):
            nloss = self.__model__(data, label)
            loss.append(nloss.detach().cpu().item())
            if idx % print_interval == 0 and rank == 0:
                sys.stdout.write("\r[Val] Reading {:d} of {:d} ".format(idx, test_loader.__len__()))
        
        if distributed:
            loss_all = [None for _ in range(0, torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(loss_all, loss)
        
        if rank == 0:
            if distributed:
                loss = loss_all[0]
                for other_loss in loss_all[1:]:
                    loss.extend(other_loss)
            
            loss = numpy.array(loss)
            L = numpy.mean(loss)
        
        return L
            
            
            







    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):

        torch.save(self.__model__.module.state_dict(), path)









    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.__model__.module.state_dict()
        loaded_state = torch.load(path, map_location="cuda:%d" % self.gpu)
        if len(loaded_state.keys()) == 1 and "model" in loaded_state:
            loaded_state = loaded_state["model"]
            newdict = {}
            delete_list = []
            for name, param in loaded_state.items():
                new_name = "__S__."+name
                newdict[new_name] = param
                delete_list.append(name)
            loaded_state.update(newdict)
            for name in delete_list:
                del loaded_state[name]
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")

                if name not in self_state:
                    print("{} is not in the model.".format(origname))
                    continue

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(), loaded_state[origname].size()))
                continue

            self_state[name].copy_(param)
