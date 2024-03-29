import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import sys
import random
import pdb
import time
import itertools
import importlib
import os

from m_DataLoader import *
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, DistributedSampler

from models.m_Models import *
import loss.m_Losses

from matplotlib import pyplot as plt

class WrappedModel(nn.Module):

    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU

    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None):
        return self.module(x, label)


class EmbedNet(nn.Module):
    # SpeakerNet: Model과 Loss Function을 같이 묶어서 처리함.

    def __init__(self, model, nPerSpeaker, **kwargs):
        super(EmbedNet, self).__init__()

        self.nPerSpeaker = nPerSpeaker
        self.__S__ = MainModel(model, **kwargs)
        self.__L__ = loss.m_Losses.LossFunction(**kwargs)

    def forward(self, data, label=None):
        # Domain Adaptation 용
        if type(data) is list:
            cut_index = data[1]
            data = data[0]
            
        data = data.reshape(-1, data.size()[-1]).cuda()
        outp = self.__S__.forward(data)

        if label == None:
            return outp

        else:
            outp = outp.reshape(self.nPerSpeaker, -1, outp.size()[-1]).transpose(1, 0).squeeze(1)
            
            # Domain Adaptation 용
            if self.__L__.trainfunc[0:2] == 'DA':
                nloss, prec = self.__L__.forward((outp, cut_index), label)
            else:
                nloss, prec = self.__L__.forward(outp, label)
            return nloss, prec


class ModelTrainer(object):
    def __init__(self, mymodel, optimizer, scheduler, gpu, **kwargs):

        self.__model__ = mymodel

        Optimizer = importlib.import_module(
            "optimizer." + optimizer).__getattribute__("Optimizer")
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)

        Scheduler = importlib.import_module(
            "scheduler." + scheduler).__getattribute__("Scheduler")
        self.__scheduler__, self.lr_step = Scheduler(
            self.__optimizer__, **kwargs)

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
        acc = 0

        # tstart = time.time()

        for sing_emb, tup_lab in loader:

            # sing_emb = sing_emb.transpose(1,0) (얘가 sampler에서 생긴 문제였을수도 ?)

            self.__model__.zero_grad()
            nloss, prec = self.__model__(sing_emb, tup_lab)
            nloss.backward()
            self.__optimizer__.step()

            loss += nloss.detach().cpu().item()
            if prec is not None:
                acc += prec.detach().cpu().item()
            counter += 1
            index += stepsize

            # telapsed = time.time() - tstart
            # tstart = time.time()

            if verbose:
                sys.stdout.write("\rProcessing {:d} of {:d}:".format(
                    index, loader.__len__() * loader.batch_size))
                sys.stdout.write(" TLoss {:f}, TAcc {:2.6f}%".format(
                    loss / counter, acc / counter))
                sys.stdout.flush()

            if self.lr_step == "iteration":
                self.__scheduler__.step()

        if self.lr_step == "epoch":
            self.__scheduler__.step()

        return (loss / counter)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====






    def validationLoss(self, valid_list, valid_path, nDataLoaderThread, distributed, print_interval=1, batch_size=1, nPerSpeaker=1, **kwargs):

        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        self.__model__.eval()

        ## Initialise trainer and data loader
        if kwargs['trainfunc'][0:2] == 'DA':
            valid_dataset = MyDistributionDataset(
                valid_list, valid_path, **kwargs)
        else:
            valid_dataset = MyDataset(
                valid_list, valid_path, nPerSpeaker=nPerSpeaker, **kwargs)

        if nPerSpeaker != 1:
            valid_sampler = train_dataset_sampler(
                data_source=valid_dataset, distributed=distributed, batch_size=1, **kwargs)
        else:
            if distributed:
                valid_sampler = DistributedSampler(valid_dataset)
            else:
                valid_sampler = None

        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                                  num_workers=nDataLoaderThread, drop_last=False, sampler=valid_sampler)

        loss = []
        prec = []
        L = 0
        P = 0

        for idx, (data, label) in enumerate(valid_loader):
            with torch.no_grad():
                nloss, prec1 = self.__model__(data, label)
            loss.append(nloss.detach().cpu().item())
            if prec1 is not None:
                prec.append(prec1.detach().cpu().item())
            if idx % print_interval == 0 and rank == 0:
                sys.stdout.write("\r[Val] Reading {:d} of {:d} ".format(
                    idx+1, valid_loader.__len__()))
                sys.stdout.flush()

        if distributed:
            loss_all = [None for _ in range(
                0, torch.distributed.get_world_size())]
            prec_all = [None for _ in range(
                0, torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(loss_all, loss)
            torch.distributed.all_gather_object(prec_all, prec)

        if rank == 0:
            if distributed:
                loss = loss_all[0]
                prec = prec_all[0]
                for other_loss in loss_all[1:]:
                    loss.extend(other_loss)
                for other_prec in prec_all[1:]:
                    prec.extend(other_prec)

            loss = numpy.array(loss)
            prec = numpy.array(prec)

            L = numpy.mean(loss)
            P = numpy.mean(prec) if len(prec) > 0 else -1

        return L, P







    def compareProcessedSingleEmbs(self, test_list, test_path, nDataLoaderThread, distributed, print_interval=1, **kwargs):

        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        self.__model__.eval()

        lines = []
        files = []
        feats = {}
        tstart = time.time()

        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()

        ## Get a list of unique file names
        files = list(itertools.chain(*[x.strip().split()[-2:] for x in lines]))
        setfiles = list(set(files))
        setfiles.sort()

        ## Define test data loader
        test_dataset = MyTestDataset(setfiles, test_path, **kwargs)

        if distributed:
            sampler = DistributedSampler(test_dataset, shuffle=False)
        else:
            sampler = None

        valid_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                  num_workers=nDataLoaderThread, drop_last=False, sampler=sampler)

        ## Extract features for every image
        for idx, data in enumerate(valid_loader):
            
            inp1 = data[0][0]
            with torch.no_grad():
                ref_feat = self.__model__(inp1).detach().cpu()
            feats[data[1][0]] = ref_feat
            telapsed = time.time() - tstart

            if idx % print_interval == 0 and rank == 0:
                sys.stdout.write("\rReading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(
                    idx+1, valid_loader.__len__(), idx / telapsed, ref_feat.size()[1]))

        all_scores = []
        all_labels = []
        all_trials = []

        if distributed:
            ## Gather features from all GPUs
            feats_all = [None for _ in range(
                0, torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(feats_all, feats)

        if rank == 0:

            tstart = time.time()
            print("")

            ## Combine gathered features
            if distributed:
                feats = feats_all[0]
                for feats_batch in feats_all[1:]:
                    feats.update(feats_batch)

            ## Read files and compute all scores
            for idx, line in enumerate(lines):

                data = line.split()

                ## Append random label if missing
                if len(data) == 2:
                    data = [random.randint(0, 1)] + data

                data[1] = data[1][:-3]+'npy'
                data[2] = data[2][:-3]+'npy'

                ref_feat = feats[data[1]].cuda()  # 512
                com_feat = feats[data[2]].cuda()  # 512

                # normaliztion after model output if loss is CS, MSE_CS
                if self.__model__.module.__L__.test_normalize:
                    ref_feat = torch.nn.functional.normalize(
                        ref_feat, p=2, dim=0)
                    com_feat = torch.nn.functional.normalize(
                        com_feat, p=2, dim=0)
                
                # L2 distance
                if kwargs['multiple_embedding_flag'] == 'B':
                    dist = torch.dist(ref_feat, com_feat, p=2).detach().cpu().numpy()

                # CS distance
                else:
                    dist = 1 - torch.nn.functional.cosine_similarity(ref_feat, com_feat, dim=0).detach().cpu().numpy()

                score = -1 * numpy.mean(dist)

                all_scores.append(score)
                all_labels.append(int(data[0]))
                all_trials.append(data[1] + " " + data[2])

                if idx % print_interval == 0:
                    telapsed = time.time() - tstart
                    sys.stdout.write(
                        "\rComputing {:d}/{:d}".format(idx+1, len(lines)))
                    # sys.stdout.flush()

        return (all_scores, all_labels, all_trials)







    def get_original_result(self, test_list, test_path, nDataLoaderThread, distributed, print_interval=1, **kwargs):

        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        lines = []
        files = []
        feats = {}
        tstart = time.time()

        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()

        ## Get a list of unique file names
        files = list(itertools.chain(*[x.strip().split()[-2:] for x in lines]))
        setfiles = list(set(files))
        setfiles.sort()

        ## Define test data loader
        test_dataset = OriginalDataset(setfiles, test_path, **kwargs)

        if distributed:
            sampler = DistributedSampler(test_dataset, shuffle=False)
        else:
            sampler = None

        valid_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                  num_workers=nDataLoaderThread, drop_last=False, sampler=sampler)

        ## Extract features for every image
        for idx, data in enumerate(valid_loader):
            ref_feat = data[0][0].cuda()
            feats[data[1][0]] = ref_feat
            telapsed = time.time() - tstart

            if idx % print_interval == 0 and rank == 0:
                sys.stdout.write(
                    "\rReading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(
                        idx+1, valid_loader.__len__(), idx / telapsed, ref_feat.size()[1])
                )

        all_scores = []
        all_labels = []
        all_trials = []

        if distributed:
            ## Gather features from all GPUs
            feats_all = [None for _ in range(
                0, torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(feats_all, feats)

        if rank == 0:

            tstart = time.time()
            print("")

            ## Combine gathered features
            if distributed:
                feats = feats_all[0]
                for feats_batch in feats_all[1:]:
                    feats.update(feats_batch)

            ## Read files and compute all scores
            for idx, line in enumerate(lines):

                data = line.split()

                ## Append random label if missing
                if len(data) == 2:
                    data = [random.randint(0, 1)] + data

                data[1] = data[1][:-3]+'npy'
                data[2] = data[2][:-3]+'npy'

                ref_feat = feats[data[1]].cuda()
                com_feat = feats[data[2]].cuda()

                # # default - 기존 방식 (distance matrix) 이용 채점
                # ref_feat = F.normalize(ref_feat, p=2, dim=1)
                # com_feat = F.normalize(com_feat, p=2, dim=1)
                # dist = torch.cdist(ref_feat, com_feat).detach().cpu().numpy()
                # score = -1 * numpy.mean(dist)

                # MySingleEmbed - 10개 평균, normalization 후 채점
                ref_feat = torch.mean(ref_feat, dim=0)
                com_feat = torch.mean(com_feat, dim=0)
                ref_feat = F.normalize(ref_feat, p=2, dim=0)
                com_feat = F.normalize(com_feat, p=2, dim=0)

                # L2 distance
                if kwargs['multiple_embedding_flag'] == 'B':
                    dist = torch.dist(ref_feat, com_feat, p=2).detach().cpu().numpy()

                # CS distance
                else:
                    dist = 1 - torch.nn.functional.cosine_similarity(ref_feat, com_feat, dim=0).detach().cpu().numpy()
                
                score = -1 * dist

                all_scores.append(score)
                all_labels.append(int(data[0]))
                all_trials.append(data[1] + " " + data[2])

                if idx % print_interval == 0:
                    telapsed = time.time() - tstart
                    sys.stdout.write("\rComputing {:d} of {:d}: {:.2f} Hz".format(
                        idx+1, len(lines), idx / telapsed))
                    sys.stdout.flush()

        return (all_scores, all_labels, all_trials)






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
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue

            self_state[name].copy_(param)
