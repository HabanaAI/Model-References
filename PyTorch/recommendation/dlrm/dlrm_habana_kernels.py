# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.


import torch
import torch.nn as nn
from habana_frameworks.torch.hpex.kernels import EmbeddingBagPreproc, EmbeddingBag
from habana_frameworks.torch.hpex.optimizers import SparseSGD, SparseAdagrad
import numpy as np
import distributed_utils
from operator import itemgetter
import inspect


optimizerValues = None

class OptimizerValues(object):
    def __init__(self, optimizer, lr):
        self._optimizer = optimizer
        self._lr        = lr

    def optimizer(self):
        return self._optimizer

    def lr(self):
        return self._lr

class EmbeddingBagSumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, indices_fwd, offsets, valid_count_fwd, kernel_mode, indices_bwd, outputRowOffsets, valid_count_bwd, weight_grad, uniqueIndices, countUniqueIndices, moments):
        # print('HabanaEmbeddingBagSumFunction:FW')
        ctx.save_for_backward(weights, indices_bwd, outputRowOffsets, valid_count_bwd, weight_grad, uniqueIndices, countUniqueIndices, moments)
        outputs = EmbeddingBag.forward(weights, indices_fwd, offsets, valid_count_fwd, kernel_mode)
        return outputs

    @staticmethod
    def backward(ctx,grad_output):
        # print('HabanaEmbeddingBagSumFunction:BW')
        weights, indices_bwd, outputRowOffsets, valid_count_bwd, weight_grad, uniqueIndices, countUniqueIndices, moments = ctx.saved_tensors
        kernel_mode = 1
        EmbeddingBag.backward(weight_grad, grad_output, indices_bwd, outputRowOffsets, valid_count_bwd, kernel_mode)
        #lr = torch.tensor([args.learning_rate]).to(device)
        global optimizerValues
        if optimizerValues is None:
            print('Optimizer not initialized')

        optimizerValues.optimizer()(
            weight_grad, weights.data, moments, uniqueIndices, optimizerValues.lr(), countUniqueIndices)

        return None, None, None, None, None, None, None, None, None, None, None, None

class HabanaEmbeddingBag(torch.nn.Module):
    def __init__(self, n, m):
        super(HabanaEmbeddingBag, self).__init__()
        # approach 1
        self.weight = nn.Parameter(torch.empty((n,m), requires_grad=True))
        self.register_buffer('weight_grad', torch.zeros_like(self.weight.data)) # To have the same buffer in case of
        self.register_buffer('moments', torch.zeros_like(self.weight.data))
        self._kernel_mode = 1

    def forward(self, indices, offsets):
        # print('HabanaEmbeddingBag; FW')
        return EmbeddingBagSumFunction.apply(self.weight, indices["indices_fwd"], offsets, indices["valid_count_fwd"], self._kernel_mode, indices["indices_bwd"], indices["outputRowOffsets"], indices["valid_count_bwd"], self.weight_grad, indices["uniqueIndices"], indices["countUniqueIndices"], self.moments)


class HabanaSparseOptimizer(torch.optim.Optimizer):
    def __init__(self, params, args):
        if args.optimizer == "sgd":
            self._SparseOpt = SparseSGD.forward
        elif args.optimizer == "adagrad":
            self._SparseOpt = SparseAdagrad.forward
        self._optimizer = args.optimizer
        if args.distributed:
            lr = args.learning_rate/args.world_size
        else:
            lr = args.learning_rate
        self._lr = torch.tensor([lr]).to('hpu')
        # Temporary hack.  Didn't find a proper way to pass the parameters to Sparse Optimizer.
        # Added Optimizer to be part of the backward gradient itself for now.
        # Passing None as gradient so step() will skip the update.
        global optimizerValues
        optimizerValues= OptimizerValues(self._SparseOpt, self._lr)
        params = list(params)
        defaults = dict(lr=lr)
        super(HabanaSparseOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # At present it should not reach here.
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['moments'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                moments = state['moments']
                #print(p.grad.to('cpu'))
                #self._SparseOpt(p.grad, p.data, moments, p.grad.indices(), lr, p.grad.idx_size())
                p.grad = None

class AllToAllAcrossDevice(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_buffer: torch.Tensor):
        output_buffer = torch.empty_like(input_buffer)
        torch.distributed.all_to_all_single(output_buffer, input_buffer)
        return output_buffer

    @staticmethod
    def backward(ctx, input_gradient):
        output_gradient = torch.empty_like(input_gradient)
        torch.distributed.all_to_all_single(output_gradient, input_gradient)
        return output_gradient

class PrintDataAcrossPass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_buffer : torch.Tensor):
        print('Forward Pass')
        print(input_buffer.to("cpu"))
        return input_buffer

    def backward(ctx, input_gradient):
        print('Backward pass')
        print(input_gradient.to("cpu"))
        return input_gradient

class CustomPreProcessor(object):
    def __init__(self, ln_emb, m_den, args, is_train=True):
        self._init_done = False
        self.num_indices_per_lookup = args.num_indices_per_lookup
        self.batch_size = args.mini_batch_size if is_train is True else args.test_mini_batch_size
        self.distributed = args.distributed
        max_size_fwd = self.num_indices_per_lookup * self.batch_size
        self.num_total_tables = len(ln_emb)
        if self.distributed:
            self.world_size = args.world_size
            self.rank = args.rank
            self.per_rank_batch_size = int(self.batch_size/self.world_size)
            if (self.batch_size % self.world_size != 0) or (len(ln_emb) < self.world_size):
                print('Unsupported batch size or table size\n')
            valid_device_emb_table, all2all_reorder = distributed_utils.dlrm_get_emb_table_map(ln_emb, args.rank, args.world_size)
            self.valid_device_emb_table = valid_device_emb_table
            #self.valid_device_emb_table = np.arange(args.rank, len(ln_emb), args.world_size)
            self.ln_emb = ln_emb[valid_device_emb_table]
        else:
            self.ln_emb = ln_emb

        self._preallocated_buffer = []
        device = torch.device('hpu')
        batch_size = self.batch_size
        if self.distributed:
            batch_size = self.per_rank_batch_size

        for table_len in self.ln_emb:
            preproc_data = {}
            preproc_data["indices_fwd"] = torch.empty([max_size_fwd], dtype=torch.int32, requires_grad = False).to(device)
            preproc_data["indices_bwd"] = torch.empty([max_size_fwd], dtype=torch.int32, requires_grad = False).to(device)
            preproc_data["uniqueIndices"] = torch.empty([max_size_fwd], dtype=torch.int32, requires_grad = False).to(device)
            preproc_data["outputRowOffsets"] = torch.empty(table_len + 1, dtype = torch.int32, requires_grad = False).to(device)
            preproc_data["coalescedGrads"] = torch.empty(table_len, dtype = torch.int32, requires_grad = False).to(device)
            preproc_data["valid_count_fwd"] = torch.empty([2], dtype = torch.int32, requires_grad = False).to(device)
            preproc_data["valid_count_bwd"] = torch.empty([2], dtype = torch.int32, requires_grad = False).to(device)
            preproc_data["countUniqueIndices"] = torch.empty([1], dtype = torch.long, requires_grad = False).to(device)
            self._preallocated_buffer.append(preproc_data)

        self.offsets = torch.empty([len(self.ln_emb), self.batch_size+1], dtype=torch.int32, requires_grad = False).to(device)
        self.X = torch.empty([batch_size, m_den], dtype=torch.float, requires_grad = False).to(device)

    def collate_habana_preprocess(self, X, lS_o, lS_i, T):
        lS_o_habana = []
        if self.distributed:
            X = X[self.rank*self.per_rank_batch_size : (self.rank+1)*self.per_rank_batch_size,]
            T = T[self.rank*self.per_rank_batch_size : (self.rank+1)*self.per_rank_batch_size,]
            #lS_o = np.take(lS_o, self.valid_device_emb_table, 0)
            lS_o = [lS_o[i] for i in range(self.num_total_tables) if i in self.valid_device_emb_table]
            lS_i = itemgetter(*self.valid_device_emb_table)(lS_i)
            if isinstance(lS_i, tuple):
                lS_i = list(lS_i)
            else:
                lS_i = [lS_i]

        for i, (idx, offset) in enumerate(zip(lS_i, lS_o)):
            idx = idx.type(torch.IntTensor)
            #offset = torch.unsqueeze(torch.cat((offset,torch.tensor([idx.numel()])),dim=0),dim=0)
            offset = torch.cat((offset, torch.tensor([idx.numel()])), dim = 0)
            offset = offset.type(torch.IntTensor)
            lS_o_habana.append(offset)

            countUniqueIndices, uniqueIndices, outputRows, outputRowOffsets = EmbeddingBagPreproc.forward(idx, offset, 4)
            valid_count_fwd = torch.tensor([offset.numel(), idx.numel()], dtype = torch.int32)
            numOffsets = countUniqueIndices.item() + 1
            valid_count_bwd = torch.tensor([numOffsets, outputRows.numel()], dtype = torch.int32)
            outputRowOffsets = torch.narrow(outputRowOffsets, 0, 0, numOffsets)
            self._preallocated_buffer[i]["indices_fwd"].copy_(idx, non_blocking=True)
            self._preallocated_buffer[i]["valid_count_fwd"].copy_(valid_count_fwd, non_blocking=True)
            self._preallocated_buffer[i]["indices_bwd"].copy_(outputRows, non_blocking=True)
            self._preallocated_buffer[i]["valid_count_bwd"].copy_(valid_count_bwd, non_blocking=True)
            self._preallocated_buffer[i]["outputRowOffsets"].copy_(outputRowOffsets, non_blocking=True)
            self._preallocated_buffer[i]["uniqueIndices"].copy_(uniqueIndices, non_blocking=True)
            self._preallocated_buffer[i]["countUniqueIndices"].copy_(countUniqueIndices, non_blocking=True)
        self.X.copy_(X, non_blocking=True)
        self.offsets.copy_(torch.stack(lS_o_habana))
        return (self.X, self.offsets, self._preallocated_buffer, T)

    def collate_wrapper_random(self, list_of_tuples):
        (X, lS_o, lS_i, T) = list_of_tuples[0]
        return self.collate_habana_preprocess(X, lS_o, lS_i, T)

    def collate_wrapper_criteo(self,list_of_tuples):
        # where each tuple is (X_int, X_cat, y)
        transposed_data = list(zip(*list_of_tuples))
        X_int = torch.log(torch.tensor(transposed_data[0], dtype=torch.float) + 1)
        X_cat = torch.tensor(transposed_data[1], dtype=torch.long)
        T = torch.tensor(transposed_data[2], dtype=torch.float32).view(-1, 1)

        batchSize = X_cat.shape[0]
        featureCnt = X_cat.shape[1]

        lS_i = [X_cat[:, i] for i in range(featureCnt)]
        lS_o = [torch.tensor(range(batchSize)) for _ in range(featureCnt)]

        return self.collate_habana_preprocess(X_int, lS_o, lS_i, T)
