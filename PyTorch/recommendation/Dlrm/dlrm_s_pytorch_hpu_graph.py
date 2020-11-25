# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import absolute_import, division, print_function, unicode_literals
import sklearn.metrics
from tricks.md_embedding_bag import PrEmbeddingBag, md_solver
from tricks.qr_embedding_bag import QREmbeddingBag
from torch.nn.parallel.scatter_gather import gather, scatter
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply
import habanaOptimizerSparseSgd_cpp
import habanaOptimizerSparseAdagrad_cpp
import preproc_cpp
import torch.nn as nn
import torch
import inspect
#import pudb
# miscellaneous
import builtins
import functools
# import bisect
# import shutil
import time
import json
# data generation
import dlrm_data_pytorch as dp
from operator import itemgetter
import distributed_utils
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# numpy
import numpy as np
import os
import sys


torch.ops.load_library(os.path.join(os.environ['BUILD_ROOT_LATEST'], "libhabana_pytorch_plugin.so"))
sys.path.insert(0, os.path.join(os.environ['BUILD_ROOT_LATEST']))

try:
    import hb_torch
except ImportError:
    assert False,"Could Not import hb_torch"

# onnx
# The onnx import causes deprecation warnings every time workers
# are spawned during testing. So, we filter out those warnings.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
# import onnx
# pytorch

# quotient-remainder trick
# mixed-dimension trick


# from torchviz import make_dot
# import torch.nn.functional as Functional
# from torch.nn.parameter import Parameter

torch.set_printoptions(precision=7)

exc = getattr(builtins, "IOError", "FileNotFoundError")


def printFnTrace(fnName):
    pass
    # print('EXEC TRACE: '+fnName )


def PDT(tensorA):
    print(tensorA.detach().cpu().numpy())

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

    @staticmethod
    def backward(ctx, input_gradient):
        print('Backward pass')
        print(input_gradient.to("cpu"))
        return input_gradient

class gv():
    countUniqueIndices = []
    uniqueIndexes = []
    outputRows = []
    outputRowOffsets = []
    outputRowOffsets_hpu = []
    coalesced_grads = []
    valid_count_bwd = []
    valid_count_fwd = []
    HabanaDlrmPreproc1 = []
    HabanaDlrmSparseOpt = None
    indices_fwd = []
    indices_bwd = []
    numEmbeddingTables = 0
    lr = torch.tensor([0.0])

# A simple hook class that returns the input and output of a layer during forward/backward pass


class Hook():
    def __init__(self, block, module, id, backward=False):
        self.name = str(block) + '_' + str(module) + '_' + str(id)
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        print('Inside HOOK for', module)
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


def printHooks(hooks, str):
    print('Printing hooks for ' + str)
    for hook in hooks:
        print('-----' * 4)
        print('Hook:{} - input'.format(hook.name))
        for x in hook.input:
            if x is None:
                print('None element')
            else:
                if isinstance(x, int) is True:
                    print(x)
                else:
                    print(x.cpu())
                    print('stride', x.stride())
                    print('size', x.size())
        print('Hook:{} - output'.format(hook.name))
        for x in hook.output:
            print(x.cpu())
            print('stride', x.stride())
            print('size', x.size())


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class HabanaDlrmPreProcFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, offsets, tableLen):
        # print('HabanaDlrmPreProcFunction; FW')
        out1, out2, out3, out4 = preproc_cpp.forward(indices, offsets, tableLen)
        return out1, out2, out3, out4

    @staticmethod
    def backward(ctx):
        # print('HabanaDlrmPreProcFunction; BW')
        return torch.ones([4, 4])


class HabanaDlrmPreProc(torch.nn.Module):
    def __init__(self, instance):
        super(HabanaDlrmPreProc, self).__init__()
        self.instance = instance
        # print('HabanaDlrmPreProc Created for instance {}'.format(self.instance))

    def forward(self, indices, offsets, tableLen):
        return HabanaDlrmPreProcFunction.apply(indices, offsets, tableLen)


class HabanaEmbeddingBag(torch.nn.Module):
    def __init__(self, table_len, embedding_size, instance):
        n = table_len
        m = embedding_size
        super(HabanaEmbeddingBag, self).__init__()
        W = np.random.uniform(
            low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
        ).astype(np.float32)
        # approach 1
        self.weight = nn.Parameter(torch.tensor(W, requires_grad=True))
        if args.optimizer == 'adagrad':
            self.moments = nn.Parameter(torch.zeros(self.weight.data.shape, requires_grad=False))
        self.instance = instance

    def forward(self, indices, offsets, valid_count_fwd, indices_bwd, offsets_bwd, valid_count_bwd, grad_weights, instance):
        output = torch.embedding_bag_sum_fwd(self.weight, indices, offsets, valid_count_fwd,
                                             indices_bwd, offsets_bwd, valid_count_bwd, grad_weights)
        return output


class HabanaOptimizerSparseSgdFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gradients, weights_in, moments_in, indices, learning_rate, valid_count):
        outputs = habanaOptimizerSparseSgd_cpp.forward(
            gradients, weights_in, moments_in, indices, learning_rate, valid_count)
        return outputs

    @staticmethod
    def backward(ctx):
        # TODO
        return torch.ones([4, 4])


class HabanaOptimizerSparseSgd(torch.nn.Module):
    def __init__(self):
        super(HabanaOptimizerSparseSgd, self).__init__()

    def forward(self, gradients, weights, moments, indices, learning_rate, valid_count):
        return HabanaOptimizerSparseSgdFunction.apply(gradients, weights, moments, indices, learning_rate, valid_count)

class HabanaOptimizerSparseAdagradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gradients, weights_in, moments_in, indices, learning_rate, valid_count):
        outputs = habanaOptimizerSparseAdagrad_cpp.forward(
            gradients, weights_in, moments_in, indices, learning_rate, valid_count)
        return outputs

    @staticmethod
    def backward(ctx):
        # TODO
        return torch.ones([4, 4])


class HabanaOptimizerSparseAdagrad(torch.nn.Module):
    def __init__(self):
        super(HabanaOptimizerSparseAdagrad, self).__init__()

    def forward(self, gradients, weights, moments, indices, learning_rate, valid_count):
        return HabanaOptimizerSparseAdagradFunction.apply(gradients, weights, moments, indices, learning_rate, valid_count)

def create_preproc(i):
    # print('Creating Preproc for instance {}'.format(i))
    gv.HabanaDlrmPreproc1.append(HabanaDlrmPreProc(i))


def create_optimizer():
    # print('Creating optimizer for instance {}'.format(i))
    if args.optimizer == "sgd":
        gv.HabanaDlrmSparseOpt = HabanaOptimizerSparseSgd()
    elif args.optimizer == "adagrad":
        gv.HabanaDlrmSparseOpt = HabanaOptimizerSparseAdagrad()


def apply_preproc(sparse_offset_group_batch, sparse_index_group_batch, i, m, ln_emb):
    printFnTrace(inspect.getframeinfo(inspect.currentframe()).function)
    # print('apply_preproc for instance {}'.format(i))
    sparse_offset_group_batch = sparse_offset_group_batch.type(torch.IntTensor)
    sparse_index_group_batch = sparse_index_group_batch.type(torch.IntTensor)

    countUniqueIndices, uniqueIndexes, gv.outputRows[i], gv.outputRowOffsets[i] = gv.HabanaDlrmPreproc1[i](
        sparse_index_group_batch, sparse_offset_group_batch, 4)

    gv.countUniqueIndices[i] = countUniqueIndices.to(device, non_blocking=True)

    # create additional tensors needed by embedding bag sum
    valid_count_fwd_cpu = torch.tensor([sparse_offset_group_batch.numel(
    ), sparse_index_group_batch.numel()], dtype=torch.int32)
    valid_count_bwd_cpu = torch.tensor(
        [countUniqueIndices.item()+1, gv.outputRows[i].numel()], dtype=torch.int32)

    numOffsets = countUniqueIndices.item()+1
    gv.outputRowOffsets[i] = torch.narrow(gv.outputRowOffsets[i], 0, 0, numOffsets)

    #Creation of static max size tensors to enable graph caching
    if (gv.indices_fwd[i].size() == torch.Size([1, 1])):
        #print("max size tensors created for instance ", i)
        max_i_size_fwd = args.num_indices_per_lookup*(sparse_offset_group_batch.numel()-1)
        gv.indices_fwd[i] = torch.empty([max_i_size_fwd],dtype=torch.int32, device = device)

        max_i_size_bwd = args.num_indices_per_lookup*(sparse_offset_group_batch.numel()-1)
        gv.indices_bwd[i] = torch.empty([max_i_size_bwd],dtype=torch.int32, device = device)

        #numoffsets_bwd = num unique indices + 1. Worst case num unique indices = ln_emb[i]
        gv.outputRowOffsets_hpu[i] = torch.empty(ln_emb[i]+1,dtype = torch.int32, device = device)
        #Max possible grad in matrix
        gv.coalesced_grads[i] = torch.empty(ln_emb[i], m, dtype=torch.float32, device = device)

        gv.valid_count_fwd[i] = torch.empty([2], dtype=torch.int32, device = device)
        gv.valid_count_bwd[i] = torch.empty([2], dtype=torch.int32, device = device)
        gv.lr = torch.tensor([args.learning_rate]).to(device, non_blocking=True)
        gv.uniqueIndexes[i] = torch.empty([max_i_size_bwd], dtype=torch.int32, device = device)

    '''
    print("emb bag instance i size", m, ln_emb[i])
    print("Tensor sizes after apply_preproc")
    print("emb instance", i)
    print("sparse_index_group_batch ", sparse_index_group_batch.size())
    print("sparse_offset_group_batch ", sparse_offset_group_batch.size())
    print("gv.indices_fwd[i] ", gv.indices_fwd[i].size())
    print("gv.outputRowOffsets_hpu[i]", gv.outputRowOffsets_hpu[i].size())
    print("gv.indices_bwd[i] ", gv.indices_bwd[i].size())
    print("gv.outputRows[i]", gv.outputRows[i].size())
    print("gv.countUniqueIndices[i]", gv.countUniqueIndices[i].size())
    print("gv.coalesced_grads[i]", gv.coalesced_grads[i].size())
    '''

    gv.indices_fwd[i].copy_(sparse_index_group_batch, non_blocking=True)
    gv.valid_count_fwd[i].copy_(valid_count_fwd_cpu, non_blocking=True)
    gv.indices_bwd[i].copy_(gv.outputRows[i], non_blocking=True)
    gv.valid_count_bwd[i].copy_(valid_count_bwd_cpu, non_blocking=True)
    gv.outputRowOffsets_hpu[i].fill_(0) #HACK
    gv.outputRowOffsets_hpu[i].copy_(gv.outputRowOffsets[i], non_blocking=True)
    gv.uniqueIndexes[i].copy_(uniqueIndexes)

def apply_optimizer_update():
    #import pudb
    for i in range(gv.numEmbeddingTables):
        weight = dlrm_habana.emb_l[i].weight.data
        if args.optimizer == 'adagrad':
            moments = dlrm_habana.emb_l[i].moments.data
        else:
            moments = torch.empty_like(weight)

        upd_emb_weights, upd_emb_moments = gv.HabanaDlrmSparseOpt(
            gv.coalesced_grads[i], weight, moments, gv.uniqueIndexes[i], gv.lr, gv.countUniqueIndices[i])

        dlrm_habana.emb_l[i].weight.data = upd_emb_weights
        if args.optimizer == 'adagrad':
            dlrm_habana.emb_l[i].moments.data = upd_emb_moments

class DLRM_Net_Habana(nn.Module):
    def create_mlp(self, ln, sigmoid_layer):
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True)

            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            # approach 2
            # LL.weight.data.copy_(torch.tensor(W))
            # LL.bias.data.copy_(torch.tensor(bt))
            # approach 3
            # LL.weight = Parameter(torch.tensor(W),requires_grad=True)
            # LL.bias = Parameter(torch.tensor(bt),requires_grad=True)
            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        # approach 1: use ModuleList
        # return layers
        # approach 2: use Sequential container to wrap all layers
        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln, valid_emb_table=None):
        emb_l = nn.ModuleList()
        create_optimizer()

        if self.ndevices > 1:
            self.valid_emb_table = valid_emb_table
            if valid_emb_table is None:
                valid_emb_table = np.arange(0, len(ln))

        for i in range(0, ln.size):
            n = ln[i]

            if self.ndevices > 1:
                # Not needed just to match with single chip run
                if i not in valid_emb_table:
                    W = np.random.uniform(
                             low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                        ).astype(np.float32)
                    continue

            # construct embedding operator
            if self.qr_flag and n > self.qr_threshold:
                EE = QREmbeddingBag(n, m, self.qr_collisions,
                                    operation=self.qr_operation, mode="sum", sparse=True)
            elif self.md_flag and n > self.md_threshold:
                _m = m[i]
                base = max(m)
                EE = PrEmbeddingBag(n, _m, base)
                # use np initialization as below for consistency...
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, _m)
                ).astype(np.float32)
                EE.embs.weight.data = torch.tensor(W, requires_grad=True)

            else:
                EE = HabanaEmbeddingBag(n, m, i)
                # print('EmbeddingBag created for config{} , instance {}'.format((n,m),i))
                create_preproc(i)
                gv.countUniqueIndices.append(torch.empty(1, 1))
                gv.uniqueIndexes.append(torch.empty(1, 1))
                gv.outputRows.append(torch.empty(1, 1))
                gv.outputRowOffsets.append(torch.empty(1, 1))
                gv.outputRowOffsets_hpu.append(torch.empty(1, 1))
                gv.coalesced_grads.append(torch.empty(1, 1))
                gv.valid_count_bwd.append(torch.empty(1, 1))
                gv.valid_count_fwd.append(torch.empty(1, 1))
                gv.indices_fwd.append(torch.empty(1, 1))
                gv.indices_bwd.append(torch.empty(1, 1))
                gv.numEmbeddingTables += 1

            emb_l.append(EE)
            # print('Total of {} EmbeddingBags Tables/Instances created= '.format(gv.numEmbeddingTables))
        # raise 1
        return emb_l

    def __init__(
        self,
        m_spa=None,
        ln_emb=None,
        ln_bot=None,
        ln_top=None,
        arch_interaction_op=None,
        arch_interaction_itself=False,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        sync_dense_params=True,
        loss_threshold=0.0,
        ndevices=-1,
        qr_flag=False,
        qr_operation="mult",
        qr_collisions=0,
        qr_threshold=200,
        md_flag=False,
        md_threshold=200,
        rank=0,
        batch_size=0,
    ):
        super(DLRM_Net_Habana, self).__init__()

        if (
            (m_spa is not None) and
            (ln_emb is not None) and
            (ln_bot is not None) and
            (ln_top is not None) and
            (arch_interaction_op is not None)
        ):

            # save arguments
            self.ndevices = ndevices
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold
            self.rank = rank
            self.batch_size = batch_size
            # create variables for QR embedding if applicable
            self.qr_flag = qr_flag
            if self.qr_flag:
                self.qr_collisions = qr_collisions
                self.qr_operation = qr_operation
                self.qr_threshold = qr_threshold
            # create variables for MD embedding if applicable
            self.md_flag = md_flag
            if self.md_flag:
                self.md_threshold = md_threshold
            self.m_spa = m_spa
            self.ln_emb = ln_emb

            # create operators
            if ndevices <= 1:
                self.emb_l = self.create_emb(m_spa, ln_emb)
            else:
                self.parallel_model_batch_size = batch_size // ndevices
                valid_device_emb_table, all2all_reorder = distributed_utils.dlrm_get_emb_table_map(ln_emb, args.rank, self.ndevices)
                if valid_device_emb_table is not None:
                    self.valid_device_emb_table = valid_device_emb_table
                else:
                    self.valid_device_emb_table = []
                self.all2all_reorder = all2all_reorder
                self.emb_l = self.create_emb(m_spa, ln_emb, valid_device_emb_table)

            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
            self.top_l = self.create_mlp(ln_top, sigmoid_top)

    def apply_mlp(self, x, layers):
        printFnTrace(inspect.getframeinfo(inspect.currentframe()).function)
        # approach 1: use ModuleList
        # for layer in layers:
        #     x = layer(x)
        # return x
        # approach 2: use Sequential container to wrap all layers
        return layers(x)

    def apply_emb(self, lS_o, lS_i, lS_vc_fwd, lS_o_bwd, lS_i_bwd, lS_vc_bwd, lS_grad_wt, emb_l):
        printFnTrace(inspect.getframeinfo(inspect.currentframe()).function)
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups

        ly = []

        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]
            #sparse_offset_group_batch_2 = torch.reshape(
            #    sparse_offset_group_batch, (sparse_offset_group_batch.size()[1],))

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            E = emb_l[k]
            valid_count_fwd = lS_vc_fwd[k]
            offsets_bwd = lS_o_bwd[k]
            indices_bwd = lS_i_bwd[k]
            valid_count_bwd = lS_vc_bwd[k]
            grad_weights = lS_grad_wt[k]

            V = E(sparse_index_group_batch, sparse_offset_group_batch,
                  valid_count_fwd, indices_bwd, offsets_bwd, valid_count_bwd, grad_weights, k)

            # print('Embedding done for k=' + str(k))
            ly.append(V)
            #print('apply_emb output', V.detach().cpu())

        return ly

    def interact_features(self, x, ly):
        printFnTrace(inspect.getframeinfo(inspect.currentframe()).function)
        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            # perform a dot product
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            # append dense feature with the interactions (into a row vector)
            # approach 1: all
            # Zflat = Z.view((batch_size, -1))
            # approach 2: unique
            _, ni, nj = Z.shape
            # approach 1: tril_indices
            # offset = 0 if self.arch_interaction_itself else -1
            # li, lj = torch.tril_indices(ni, nj, offset=offset)
            # approach 2: custom
            offset = 1 if self.arch_interaction_itself else 0
            lij = torch.tensor([i*nj+j for i in range(nj) for j in range(i + offset)])
            lij_hpu = lij.to(device)
            Z_temp = Z.view(batch_size,ni*nj)
            Zflat = torch.index_select(Z_temp, 1, lij_hpu)

            # concatenate dense features and interactions
            R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            # pudb.set_trace()
            # concatenation features (into a row vector)
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )

        return R

    def forward(self, dense_x, lS_o, lS_i, lS_vc_fwd, lS_o_bwd, lS_i_bwd, lS_vc_bwd, lS_grad_wt):
        printFnTrace(inspect.getframeinfo(inspect.currentframe()).function)
        if self.ndevices <= 1:
            return self.sequential_forward(dense_x, lS_o, lS_i, lS_vc_fwd, lS_o_bwd, lS_i_bwd, lS_vc_bwd, lS_grad_wt)
        else:
            return self.parallel_forward(dense_x, lS_o, lS_i, lS_vc_fwd, lS_o_bwd, lS_i_bwd, lS_vc_bwd, lS_grad_wt)

    def sequential_forward(self, dense_x, lS_o, lS_i, lS_vc_fwd, lS_o_bwd, lS_i_bwd, lS_vc_bwd, lS_grad_wt):
        printFnTrace(inspect.getframeinfo(inspect.currentframe()).function)
        # process dense features (using bottom mlp), resulting in a row vector
        x = self.apply_mlp(dense_x, self.bot_l)

        ly = self.apply_emb(lS_o, lS_i, lS_vc_fwd, lS_o_bwd, lS_i_bwd, lS_vc_bwd, lS_grad_wt, self.emb_l)
        # print('ly')
        # for y in ly:
        #      print(y.detach().cpu().numpy())

        # interact features (dense and sparse)
        z = self.interact_features(x, ly)
        #print("z_interact: ", z.detach().cpu().numpy())

        # obtain probability of a click (using top mlp)
        p = self.apply_mlp(z, self.top_l)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        else:
            z = p


        #print("z_return: ", z.detach().cpu().numpy())
        return z

    def exchange_emb(self, ly, batch_size, valid_device_emb_table):
        max_table_per_device = (len(self.ln_emb) + self.ndevices - 1) // self.ndevices
        # TBD: issue with slice copy on device, so use cpu for slicing and copy
        exchange_input_buffer = torch.zeros(batch_size*self.ndevices, max_table_per_device*self.m_spa, device="cpu", dtype = ly[0].dtype)
        for i in range(len(ly)):
            exchange_input_buffer[:,i*self.m_spa:(i+1)*self.m_spa] = ly[i]
        #exchange_input_buffer = PrintDataAcrossPass.apply(exchange_input_buffer)
        exchange_input_buffer = exchange_input_buffer.to(ly[0].device)
        exchange_output_buffer = AllToAllAcrossDevice.apply(exchange_input_buffer)
        exchange_output_buffer = exchange_output_buffer.to("cpu")
        #exchange_output_buffer = PrintDataAcrossPass.apply(exchange_output_buffer)
        rank_specific_data = []
        for i in range(self.ndevices):
            rank_specific_data.append(exchange_output_buffer[i*batch_size:(i+1)*batch_size,:])
        n = len(self.ln_emb) % self.ndevices
        if n == 0:
            n = self.ndevices
        out_ly = []
        for i in range(n):
            for j in range(max_table_per_device):
                out_ly.append(rank_specific_data[i][:,j*self.m_spa:(j+1)*self.m_spa])
        for i in range(n, self.ndevices):
            for j in range(max_table_per_device-1):
                out_ly.append(rank_specific_data[i][:,j*self.m_spa:(j+1)*self.m_spa])
        # Needed to match with single chip
        out_ly = [out_ly[pos].to(ly[0].device) for pos in self.all2all_reorder]
        return out_ly

    def parallel_forward(self, dense_x, lS_o, lS_i, lS_vc_fwd, lS_o_bwd, lS_i_bwd, lS_vc_bwd, lS_grad_wt):
        ### prepare model (overwrite) ###
        # WARNING: # of devices must be >= batch size in parallel_forward call
        #batch_size = dense_x.size()[0]
        batch_size = self.parallel_model_batch_size
        ndevices = min(self.ndevices, batch_size, len(self.emb_l))
        device_ids = range(ndevices)
        # WARNING: must redistribute the model if mini-batch size changes(this is common
        # for last mini-batch, when # of elements in the dataset/batch size is not even

        '''
        if self.parallel_model_batch_size != batch_size:
            self.parallel_model_is_not_prepared = True

        if self.parallel_model_is_not_prepared or self.sync_dense_params:
            # replicate mlp (data parallelism)
            self.bot_l_replicas = replicate(self.bot_l, device_ids)
            self.top_l_replicas = replicate(self.top_l, device_ids)
            self.parallel_model_batch_size = batch_size

        if self.parallel_model_is_not_prepared:
            # distribute embeddings (model parallelism)
            t_list = []
            for k, emb in enumerate(self.emb_l):
                d = torch.device("cuda:" + str(k % ndevices))
                emb.to(d)
                t_list.append(emb.to(d))
            self.emb_l = nn.ModuleList(t_list)
            self.parallel_model_is_not_prepared = False

        ### prepare input (overwrite) ###
        # scatter dense features (data parallelism)
        # print(dense_x.device)
        dense_x = scatter(dense_x, device_ids, dim=0)
        '''
        # distribute sparse features (model parallelism)
        if (len(self.emb_l) != len(lS_o)) or (len(self.emb_l) != len(lS_i)):
            sys.exit("ERROR: corrupted model input detected in parallel_forward call")

        '''
        t_list = []
        i_list = []
        for k, _ in enumerate(self.emb_l):
            d = torch.device("cuda:" + str(k % ndevices))
            t_list.append(lS_o[k].to(d))
            i_list.append(lS_i[k].to(d))
        lS_o = t_list
        lS_i = i_list

        ### compute results in parallel ###
        # bottom mlp
        # WARNING: Note that the self.bot_l is a list of bottom mlp modules
        # that have been replicated across devices, while dense_x is a tuple of dense
        # inputs that has been scattered across devices on the first (batch) dimension.
        # The output is a list of tensors scattered across devices according to the
        # distribution of dense_x.
        x = parallel_apply(self.bot_l_replicas, dense_x, None, device_ids)
        # debug prints
        # print(x)

        '''
        x = self.apply_mlp(dense_x, self.bot_l)
        # embeddings
        ly = self.apply_emb(lS_o, lS_i, lS_vc_fwd, lS_o_bwd, lS_i_bwd, lS_vc_bwd, lS_grad_wt, self.emb_l)
        # debug prints
        # print(ly)

        # butterfly shuffle (implemented inefficiently for now)
        # WARNING: Note that at this point we have the result of the embedding lookup
        # for the entire batch on each device. We would like to obtain partial results
        # corresponding to all embedding lookups, but part of the batch on each device.
        # Therefore, matching the distribution of output of bottom mlp, so that both
        # could be used for subsequent interactions on each device.
        if len(self.emb_l) != len(ly):
            sys.exit("ERROR: corrupted intermediate result in parallel_forward call")

        '''
        t_list = []
        for k, _ in enumerate(self.emb_l):
            d = torch.device("cuda:" + str(k % ndevices))
            y = scatter(ly[k], device_ids, dim=0)
            t_list.append(y)
        # adjust the list to be ordered per device
        ly = list(map(lambda y: list(y), zip(*t_list)))
        # debug prints
        # print(ly)
        '''
        ly = self.exchange_emb(ly, batch_size, valid_device_emb_table)
        z0 = self.interact_features(x, ly)
        p0 = self.apply_mlp(z0, self.top_l)

        '''
        # interactions
        z = []
        for k in range(ndevices):
            zk = self.interact_features(x[k], ly[k])
            z.append(zk)
        # debug prints
        # print(z)

        # top mlp
        # WARNING: Note that the self.top_l is a list of top mlp modules that
        # have been replicated across devices, while z is a list of interaction results
        # that by construction are scattered across devices on the first (batch) dim.
        # The output is a list of tensors scattered across devices according to the
        # distribution of z.
        p = parallel_apply(self.top_l_replicas, z, None, device_ids)

        ### gather the distributed results ###
        p0 = gather(p, self.output_d, dim=0)

        '''
        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z0 = torch.clamp(
                p0, min=self.loss_threshold, max=(1.0 - self.loss_threshold)
            )
        else:
            z0 = p0

        return z0

    def printParamsAndGrads(self, grads=True):
        print('Printing Params')
        [print(name, p.cpu(), p.grad_fn, p.cpu().grad_fn) for name, p in self.named_parameters()]
        if grads is True:
            print('Printing Params grads')
            [print(name, p.grad.cpu()) for name, p in self.named_parameters()]
        # print(vars(gv))

def flatten_tensor(params):
    tensor_list = []
    for param in params:
        if (param.requires_grad):
            tensor_list.append(param.grad)
    flat = torch.cat([t.contiguous().view(-1) for t in tensor_list], dim=0)
    return flat, tensor_list

def unflatten_tensor(flat, tensor_list):
    outputs = []
    offset = 0
    for tensor in tensor_list:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return outputs

def update_tensors(params, outputs):
    idx=0
    for tparam in params:
        if (tparam.requires_grad):
            #print("outputs :: ",outputs1[idx].to("cpu"))
            tparam.grad.copy_(outputs[idx])
            idx+=1
    return outputs

if __name__ == "__main__":
    ### import packages ###
    import sys
    import argparse

    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    # model related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
    parser.add_argument("--arch-embedding-size", type=str, default="4-3-2")

    # j will be replaced with the table number
    parser.add_argument("--arch-mlp-bot", type=str, default="4-3-2")
    parser.add_argument("--arch-mlp-top", type=str, default="4-2-1")
    parser.add_argument("--arch-interaction-op", type=str, default="cat")
    parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
    # embedding table options
    parser.add_argument("--md-flag", action="store_true", default=False)
    parser.add_argument("--md-threshold", type=int, default=200)
    parser.add_argument("--md-temperature", type=float, default=0.3)
    parser.add_argument("--md-round-dims", action="store_true", default=False)
    parser.add_argument("--qr-flag", action="store_true", default=False)
    parser.add_argument("--qr-threshold", type=int, default=200)
    parser.add_argument("--qr-operation", type=str, default="mult")
    parser.add_argument("--qr-collisions", type=int, default=4)
    # activations and loss
    parser.add_argument("--activation-function", type=str, default="relu")
    parser.add_argument("--loss-function", type=str, default="bce")  # mse or bce or wbce
    parser.add_argument("--loss-weights", type=str, default="1.0-1.0")  # for wbce
    parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7
    parser.add_argument("--round-targets", type=bool, default=False)
    # data
    parser.add_argument("--data-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument(
        "--data-generation", type=str, default="random"
    )  # synthetic or dataset
    parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
    parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--memory-map", action="store_true", default=False)
    # training
    parser.add_argument("--mini-batch-size", type=int, default=4)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--sync-dense-params", type=bool, default=True)
    parser.add_argument("--optimizer", type=str, default="sgd")
    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)
    # onnx
    parser.add_argument("--save-onnx", action="store_true", default=False)
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=False)
    # debugging and profiling
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--test-mini-batch-size", type=int, default=-1)
    parser.add_argument("--test-num-workers", type=int, default=-1)
    parser.add_argument("--print-time", action="store_true", default=False)
    parser.add_argument("--debug-mode", action="store_true", default=False)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    parser.add_argument("--plot-compute-graph", action="store_true", default=False)
    # store/load model
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    # mlperf logging (disables other output and stops early)
    parser.add_argument("--mlperf-logging", action="store_true", default=False)
    # stop at target accuracy Kaggle 0.789, Terabyte (sub-sampled=0.875) 0.8107
    parser.add_argument("--mlperf-acc-threshold", type=float, default=0.0)
    # stop at target AUC Terabyte (no subsampling) 0.8025
    parser.add_argument("--mlperf-auc-threshold", type=float, default=0.0)
    parser.add_argument("--mlperf-bin-loader", action='store_true', default=False)
    parser.add_argument("--mlperf-bin-shuffle", action='store_true', default=False)
    parser.add_argument('--no-habana', action='store_true', default=False,
                        help='disables habana training')
    parser.add_argument("--use-jit-trace", action='store_true', default=True,
                        help='run with torch jit trace mode')
    parser.add_argument('--hmp', dest='is_hmp', action='store_true', help='enable hmp mode')
    parser.add_argument('--hmp-bf16', default='', help='path to bf16 ops list in hmp O1 mode')
    parser.add_argument('--hmp-fp32', default='', help='path to fp32 ops list in hmp O1 mode')
    parser.add_argument('--hmp-opt-level', default='O1', help='choose optimization level for hmp')
    parser.add_argument('--hmp-verbose', action='store_true', help='enable verbose mode for hmp')
    parser.add_argument("--distributed", action="store_true", default=False)
    parser.add_argument("--use-lazy-eval", action='store_true', default=False,
                        help='not enabled yet :run with lazy eval mode')

    args = parser.parse_args()

    if args.is_hmp:
        from hmp import hmp
        hmp.convert(opt_level=args.hmp_opt_level, bf16_file_path=args.hmp_bf16,
                    fp32_file_path=args.hmp_fp32, isVerbose=args.hmp_verbose)

    use_hpu = not args.no_habana

    print(args)
    if args.mlperf_logging:
        print('command line args: ', json.dumps(vars(args)))

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)
    torch.set_printoptions(precision=args.print_precision)
    torch.manual_seed(args.numpy_rand_seed)
    print('MB= {}, DataSize={}'.format(args.mini_batch_size, args.data_size))

    if (args.test_mini_batch_size < 0):
        # if the parameter is not set, use the training batch size
        args.test_mini_batch_size = args.mini_batch_size
    if (args.test_num_workers < 0):
        # if the parameter is not set, use the same parameter for training
        args.test_num_workers = args.num_workers

    if args.distributed:
        print("Start configuring")
        distributed_utils.init_distributed_mode(args)

    use_gpu = args.use_gpu and torch.cuda.is_available()
    if use_gpu:
        torch.cuda.manual_seed_all(args.numpy_rand_seed)
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda", 0)
        ngpus = torch.cuda.device_count()  # 1
        print("Using {} GPU(s)...".format(ngpus))
    elif use_hpu:
        torch.ops.load_library(os.path.join(os.environ['BUILD_ROOT_LATEST'], "libhabana_pytorch_plugin.so"))
        device = torch.device('habana')
        sys.path.insert(0, os.path.join(os.environ['BUILD_ROOT_LATEST']))
        print('Using HPU...')
    else:
        device = torch.device("cpu")
        print("Using CPU...")

    ### prepare training data ###
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    # input data
    if (args.data_generation == "dataset"):

        train_data, train_ld, test_data, test_ld = \
            dp.make_criteo_data_and_loaders(args)
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
        nbatches_test = len(test_ld)

        ln_emb = train_data.counts
        # enforce maximum limit on number of vectors per embedding
        if args.max_ind_range > 0:
            ln_emb = np.array(list(map(
                lambda x: x if x < args.max_ind_range else args.max_ind_range,
                ln_emb
            )))
        m_den = train_data.m_den
        ln_bot[0] = m_den
    else:
        # input and target at random
        ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
        m_den = ln_bot[0]
        train_data, train_ld = dp.make_random_data_and_loader(args, ln_emb, m_den)
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)

    ### parse command line arguments ###
    m_spa = args.arch_sparse_feature_size
    num_fea = ln_emb.size + 1  # num sparse + num dense features
    m_den_out = ln_bot[ln_bot.size - 1]
    if args.arch_interaction_op == "dot":
        # approach 1: all
        # num_int = num_fea * num_fea + m_den_out
        # approach 2: unique
        if args.arch_interaction_itself:
            num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
        else:
            num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    elif args.arch_interaction_op == "cat":
        num_int = num_fea * m_den_out
    else:
        sys.exit(
            "ERROR: --arch-interaction-op=" +
            args.arch_interaction_op +
            " is not supported"
        )
    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

    # sanity check: feature sizes and mlp dimensions must match
    if m_den != ln_bot[0]:
        sys.exit(
            "ERROR: arch-dense-feature-size " +
            str(m_den) +
            " does not match first dim of bottom mlp " +
            str(ln_bot[0])
        )
    if args.qr_flag:
        if args.qr_operation == "concat" and 2 * m_spa != m_den_out:
            sys.exit(
                "ERROR: 2 arch-sparse-feature-size " +
                str(2 * m_spa) +
                " does not match last dim of bottom mlp " +
                str(m_den_out) +
                " (note that the last dim of bottom mlp must be 2x the embedding dim)"
            )
        if args.qr_operation != "concat" and m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size " +
                str(m_spa) +
                " does not match last dim of bottom mlp " +
                str(m_den_out)
            )
    else:
        if m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size " +
                str(m_spa) +
                " does not match last dim of bottom mlp " +
                str(m_den_out)
            )
    if num_int != ln_top[0]:
        sys.exit(
            "ERROR: # of feature interactions " +
            str(num_int) +
            " does not match first dimension of top mlp " +
            str(ln_top[0])
        )

    # assign mixed dimensions if applicable
    if args.md_flag:
        m_spa = md_solver(
            torch.tensor(ln_emb),
            args.md_temperature,  # alpha
            d0=m_spa,
            round_dim=args.md_round_dims
        ).tolist()

    # test prints (model arch)
    if args.debug_mode:
        print("model arch:")
        print(
            "mlp top arch " +
            str(ln_top.size - 1) +
            " layers, with input to output dimensions:"
        )
        print(ln_top)
        print("# of interactions")
        print(num_int)
        print(
            "mlp bot arch " +
            str(ln_bot.size - 1) +
            " layers, with input to output dimensions:"
        )
        print(ln_bot)
        print("# of features (sparse and dense)")
        print(num_fea)
        print("dense feature size")
        print(m_den)
        print("sparse feature size")
        print(m_spa)
        print(
            "# of embeddings (= # of sparse features) " +
            str(ln_emb.size) +
            ", with dimensions " +
            str(m_spa) +
            "x:"
        )
        print(ln_emb)

        print("data (inputs and targets):")
        for j, (X, lS_o, lS_i, T) in enumerate(train_ld):
            # early exit if nbatches was set by the user and has been exceeded
            if nbatches > 0 and j >= nbatches:
                break

            print("mini-batch: %d" % j)
            print(X.detach().cpu().numpy())
            # transform offsets to lengths when printing
            print(
                [
                    np.diff(
                        S_o.detach().cpu().tolist() + list(lS_i[i].shape)
                    ).tolist()
                    for i, S_o in enumerate(lS_o)
                ]
            )
            print([S_i.detach().cpu().tolist() for S_i in lS_i])
            print(T.detach().cpu().numpy())

    if args.distributed:
        ndevices = args.world_size
        rank = args.rank
        world_size = args.world_size
    else:
        rank = 0
        world_size = 1
        if use_gpu and not use_hpu:
            ndevices = min(ngpus, args.mini_batch_size, num_fea - 1)
        else:
            ndevices = -1

    ### construct the neural network specified above ###
    # WARNING: to obtain exactly the same initialization for
    # the weights we need to start from the same random seed.
    # np.random.seed(args.numpy_rand_seed)

    dlrm_habana = DLRM_Net_Habana(
        m_spa,
        ln_emb,
        ln_bot,
        ln_top,
        arch_interaction_op=args.arch_interaction_op,
        arch_interaction_itself=args.arch_interaction_itself,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        sync_dense_params=args.sync_dense_params,
        loss_threshold=args.loss_threshold,
        ndevices=ndevices,
        qr_flag=args.qr_flag,
        qr_operation=args.qr_operation,
        qr_collisions=args.qr_collisions,
        qr_threshold=args.qr_threshold,
        md_flag=args.md_flag,
        md_threshold=args.md_threshold,
        rank=rank,
        batch_size=args.mini_batch_size

    )
    # print('Net at Init')
    # dlrm_habana.printParamsAndGrads(grads=False)

    # hookF = []
    # hookB = []

    # for layer in list(dlrm_habana._modules.items()):
    #     if isinstance(layer[1],nn.Sequential):
    #         print('found Sequential '+'----'*3)
    #         i=0
    #         for a in enumerate(layer[1]):
    #             print('Attaching hook to {} module: {}'.format(i,a))
    #             hookF.append(Hook(layer[0],a[1],a[0]))
    #             hookB.append(Hook(layer[0],a[1],a[0],backward=True))
    #             i += 1

    #     if isinstance(layer[1],nn.ModuleList):
    #         print('found module list'+'----'*3)
    #         i=0
    #         for a in enumerate(layer[1]):
    #             print('Attaching hook to {} module: {}'.format(i,a))
    #             hookF.append(Hook(layer[0],a[1],a[0]))
    #             hookB.append(Hook(layer[0],a[1],a[0],backward=True))
    #             i += 1

    # test prints
    if args.debug_mode:
        print("initial parameters (weights and bias):")
        for param in dlrm.parameters():
            print(param.detach().cpu().numpy())
        # print(dlrm)
        for param in dlrm_habana.parameters():
            print(param.detach().cpu().numpy())

    if use_gpu:
        # Custom Model-Data Parallel
        # the mlps are replicated and use data parallelism, while
        # the embeddings are distributed and use model parallelism
        dlrm = dlrm.to(device)  # .cuda()
        if dlrm.ndevices > 1:
            dlrm.emb_l = dlrm.create_emb(m_spa, ln_emb)

    if use_hpu:
        # dlrm = dlrm.to(device)
        dlrm_habana = dlrm_habana.to(device)
        if dlrm_habana.ndevices > 1:
            # enable ddp hooks only during lazy/eager mode since
            # traced DDP hooks are missed post first iteration
            if args.use_lazy_eval:
                dlrm_habana.bot_l = DDP(dlrm_habana.bot_l, bucket_cap_mb=8192)
                dlrm_habana.top_l = DDP(dlrm_habana.top_l, bucket_cap_mb=8192)
            valid_device_emb_table, _ = distributed_utils.dlrm_get_emb_table_map(ln_emb, args.rank, args.world_size)
            if valid_device_emb_table is None:
                valid_device_emb_table = []

    # specify the loss function
    if args.loss_function == "mse":
        loss_fn = torch.nn.MSELoss(reduction="mean")
    elif args.loss_function == "bce":
        loss_fn = torch.nn.BCELoss(reduction="mean")
    elif args.loss_function == "wbce":
        loss_ws = torch.tensor(np.fromstring(args.loss_weights, dtype=float, sep="-"))
        loss_fn = torch.nn.BCELoss(reduction="none")
    else:
        sys.exit("ERROR: --loss-function=" + args.loss_function + " is not supported")

    if args.distributed and not args.use_lazy_eval:
        #TBD: need to use the right scaled LR for multinode
        slr = args.learning_rate / world_size
    else:
        slr = args.learning_rate

    if not args.inference_only:
        # specify the optimizer algorithm
        if args.optimizer == "sgd":
            optimizer = torch.optim.SGD(list(dlrm_habana.top_l.parameters())
                                    + list(dlrm_habana.bot_l.parameters()), lr=slr)
        elif args.optimizer == "adagrad":
            optimizer = torch.optim.Adagrad(list(dlrm_habana.top_l.parameters())
                                    + list(dlrm_habana.bot_l.parameters()), lr=slr)
        else:
            sys.exit("ERROR: --optimizer=" + args.optimizer + " is not supported")

    ### main loop ###

    def time_wrap(use_gpu):
        if use_gpu:
            torch.cuda.synchronize()
        return time.time()

    def enable_tracing(model, X_device, lS_o, lS_i, lS_vc_fwd, lS_o_bwd, lS_i_bwd, lS_vc_bwd, lS_grad_wt):
        torch._C._debug_set_autodiff_subgraph_inlining(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        hb_torch.enable()
        dlrm_trace = torch.jit.trace(model, (X_device, lS_o, lS_i, lS_vc_fwd, lS_o_bwd,
                                             lS_i_bwd, lS_vc_bwd, lS_grad_wt), check_trace=False)
        # print("tracing completed")
        #print(dlrm_trace.graph_for(X_device, lS_o, lS_i, lS_vc_fwd, lS_o_bwd,
        #                           lS_i_bwd, lS_vc_bwd, lS_grad_wt))
        return dlrm_trace

    def dlrm_habana_wrap(model, X_device, lS_o, lS_i, lS_vc_fwd, lS_o_bwd, lS_i_bwd, lS_vc_bwd, lS_grad_wt):
        printFnTrace(inspect.getframeinfo(inspect.currentframe()).function)
        return model(X_device,
                     lS_o,
                     lS_i,
                     lS_vc_fwd,
                     lS_o_bwd,
                     lS_i_bwd,
                     lS_vc_bwd,
                     lS_grad_wt
                     )

    import inspect

    def loss_fn_wrap(Z, T, use_gpu, use_hpu, device):
        printFnTrace(inspect.getframeinfo(inspect.currentframe()).function)
        if args.loss_function == "mse" or args.loss_function == "bce":
            if use_gpu or use_hpu:
                return loss_fn(Z, T.to(device))
            else:
                return loss_fn(Z, T)
        elif args.loss_function == "wbce":
            if use_gpu:
                loss_ws_ = loss_ws[T.data.view(-1).long()].view_as(T).to(device)
                loss_fn_ = loss_fn(Z, T.to(device))
            else:
                loss_ws_ = loss_ws[T.data.view(-1).long()].view_as(T)
                loss_fn_ = loss_fn(Z, T.to(device))
            loss_sc_ = loss_ws_ * loss_fn_
            return loss_sc_.mean()

    def loss_fn_habana_wrap(Z, T, use_gpu, use_hpu, device):
        printFnTrace(inspect.getframeinfo(inspect.currentframe()).function)
        if args.loss_function == "mse" or args.loss_function == "bce":
            if use_gpu or use_hpu:
                return loss_fn(Z, T.to(device))
            else:
                return loss_fn(Z, T)
        elif args.loss_function == "wbce":
            if use_gpu:
                loss_ws_ = loss_ws[T.data.view(-1).long()].view_as(T).to(device)
                loss_fn_ = loss_fn(Z, T.to(device))
            else:
                loss_ws_ = loss_ws[T.data.view(-1).long()].view_as(T)
                loss_fn_ = loss_fn(Z, T.to(device))
            loss_sc_ = loss_ws_ * loss_fn_
            # debug prints
            # print(loss_ws_)
            # print(loss_fn_)
            return loss_sc_.mean()

    # training or inference
    best_gA_test = 0
    best_auc_test = 0
    skip_upto_epoch = 0
    skip_upto_batch = 0
    total_time = 0
    total_loss = 0
    total_accu = 0
    total_iter = 0
    total_samp = 0
    k = 0

    # Load model is specified
    if not (args.load_model == ""):
        print("Loading saved model {}".format(args.load_model))
        if use_gpu:
            if dlrm.ndevices > 1:
                # NOTE: when targeting inference on multiple GPUs,
                # load the model as is on CPU or GPU, with the move
                # to multiple GPUs to be done in parallel_forward
                ld_model = torch.load(args.load_model)
            else:
                # NOTE: when targeting inference on single GPU,
                # note that the call to .to(device) has already happened
                ld_model = torch.load(
                    args.load_model,
                    map_location=torch.device('cuda')
                    # map_location=lambda storage, loc: storage.cuda(0)
                )
        else:
            # when targeting inference on CPU
            ld_model = torch.load(args.load_model, map_location=torch.device('cpu'))
        dlrm_habana.load_state_dict(ld_model["state_dict"])
        ld_j = ld_model["iter"]
        ld_k = ld_model["epoch"]
        ld_nepochs = ld_model["nepochs"]
        ld_nbatches = ld_model["nbatches"]
        ld_nbatches_test = ld_model["nbatches_test"]
        ld_gA = ld_model["train_acc"]
        ld_gL = ld_model["train_loss"]
        ld_total_loss = ld_model["total_loss"]
        ld_total_accu = ld_model["total_accu"]
        ld_gA_test = ld_model["test_acc"]
        ld_gL_test = ld_model["test_loss"]
        if not args.inference_only:
            # optimizer.load_state_dict(ld_model["opt_state_dict"])
            best_gA_test = ld_gA_test
            total_loss = ld_total_loss
            total_accu = ld_total_accu
            skip_upto_epoch = ld_k  # epochs
            skip_upto_batch = ld_j  # batches
        else:
            args.print_freq = ld_nbatches
            args.test_freq = 0

        print(
            "Saved at: epoch = {:d}/{:d}, batch = {:d}/{:d}, ntbatch = {:d}".format(
                ld_k, ld_nepochs, ld_j, ld_nbatches, ld_nbatches_test
            )
        )
        print(
            "Training state: loss = {:.6f}, accuracy = {:3.3f} %".format(
                ld_gL, ld_gA * 100
            )
        )
        print(
            "Testing state: loss = {:.6f}, accuracy = {:3.3f} %".format(
                ld_gL_test, ld_gA_test * 100
            )
        )

    print("time/loss/accuracy (if enabled):")
    print('skip_upto_epoch=',skip_upto_epoch)
    print('skip_upto_batch=',skip_upto_batch)

    training_resumed = False
    with torch.autograd.profiler.profile(args.enable_profiling, use_gpu) as prof:
        model_to_run = dlrm_habana
        is_first_it = True
        while k < args.nepochs:
            if k < skip_upto_epoch:
                print('skipping epoch')
                continue

            accum_time_begin = time_wrap(use_gpu)

            if args.mlperf_logging:
                previous_iteration_time = None

            for j, (X, lS_o, lS_i, T) in enumerate(train_ld):

                #print("lS_i python", lS_i)
                # np.random.seed(args.numpy_rand_seed)
                # torch.manual_seed(args.numpy_rand_seed)

                if j < skip_upto_batch and not(training_resumed):
                    if j == (skip_upto_batch-1):
                        training_resumed = True
                        print('Last batch of resumed epoch')
                    print('skipping batch')
                    continue
                training_resumed = True

                # early exit if batch is partial
                if X.size()[0] < args.mini_batch_size:
                    print('Breaking out of the epoch as  batch was partial. Number of samples:',X.size()[0])
                    break

                if args.distributed:
                    # mini batch size is expected to be a multiple of world_size
                    train_batch_size = int(X.size()[0]/args.world_size)
                    X = np.take(X,np.arange(args.rank*train_batch_size,(args.rank+1)*train_batch_size), 0)
                    if valid_device_emb_table is not None:
                        lS_o = np.take(lS_o, valid_device_emb_table, 0)
                        lS_i = itemgetter(*valid_device_emb_table)(lS_i)
                    else:
                        lS_o = []
                        lS_i = []
                    T = T[args.rank*train_batch_size:(args.rank+1)*train_batch_size]
                    if isinstance(lS_i, tuple):
                        lS_i = list(lS_i)
                    else:
                        lS_i = [lS_i]

                # embedding bag on Habana needs the last offset to be additionally appended which is pointing to end of indices
                lS2_o_scratch = []
                for kk in range(len(lS_i)):
                    lS2_o_scratch.append(torch.unsqueeze(
                        torch.cat((lS_o[kk], torch.tensor([lS_i[kk].numel()])), dim=0), dim=0))

                lS_o = torch.cat(tuple(lS2_o_scratch), dim=0)
                #print('lS_o after updating for last offset',lS_o)
                #print("ls_i: ", lS_i)

                with torch.no_grad():
                    for kk, sparse_index_group_batch in enumerate(lS_i):
                        sparse_offset_group_batch = lS_o[kk]
                        if not args.distributed:
                            apply_preproc(sparse_offset_group_batch, sparse_index_group_batch, kk, m_spa, ln_emb)
                        else:
                            apply_preproc(sparse_offset_group_batch, sparse_index_group_batch, kk, m_spa, ln_emb[valid_device_emb_table])

                lS2_o = [S_o.to(torch.int32).to(device, non_blocking=True) for S_o in lS_o] if isinstance(lS_o, list) \
                    else lS_o.to(torch.int32).to(device, non_blocking=True)

                X_device = X.to(device, non_blocking=True)

                if args.use_jit_trace and is_first_it:
                    model_to_run = enable_tracing(dlrm_habana, X_device, lS2_o, gv.indices_fwd,
                                                  gv.valid_count_fwd, gv.outputRowOffsets_hpu, gv.indices_bwd, gv.valid_count_bwd, gv.coalesced_grads)

                # we are not using DDP wrapper now, so explicitly broadcast the params
                if args.distributed and is_first_it and not args.use_lazy_eval:
                    for tparam in model_to_run.top_l.parameters():
                        torch.distributed.broadcast(tparam, 0)
                    for bparam in model_to_run.bot_l.parameters():
                        torch.distributed.broadcast(bparam, 0)

                is_first_it = False


                if args.mlperf_logging:
                    current_time = time_wrap(use_gpu)
                    if previous_iteration_time:
                        iteration_time = current_time - previous_iteration_time
                    else:
                        iteration_time = 0
                    previous_iteration_time = current_time
                else:
                    t1 = time_wrap(use_gpu)

                # early exit if nbatches was set by the user and has been exceeded
                if nbatches > 0 and j >= nbatches:
                    print('Breaking out of the epoch as j>=nbatches')
                    break

                # forward pass
                Z_habana = dlrm_habana_wrap(model_to_run,  X_device, lS2_o, gv.indices_fwd,
                                                  gv.valid_count_fwd, gv.outputRowOffsets_hpu, gv.indices_bwd, gv.valid_count_bwd, gv.coalesced_grads)

                #print("Z_habana", Z_habana.cpu())

                if args.use_jit_trace and is_first_it:
                    print('Z_habana.grad_fn:',Z_habana.grad_fn)
                # print('Printing F hooks')
                # for hook in hookF:
                #     print(hook.input.cpu().numpy())
                #     print(hook.output.cpu().numpy())
                #     print('---'*17)

                # print('Printing B hooks')
                # for hook in hookB:
                #     print(hook.input)
                #     print(hook.output)
                #     print('---'*17)
                # loss
                # print(T)
                E_habana = loss_fn_wrap(Z_habana, T, use_gpu, use_hpu, device)

                # compute loss and accuracy
                L_habana = E_habana.detach().cpu().numpy()  # numpy array
                S_habana = Z_habana.float().detach().cpu().numpy()  # numpy array
                T = T.detach().cpu().numpy()  # numpy array
                mbs = T.shape[0]  # = args.mini_batch_size except maybe for last
                A = np.sum((np.round(S_habana, 0) == T).astype(np.uint8))

                if not args.inference_only:
                    # scaled error gradient propagation
                    # (where we do not accumulate gradients across mini-batches)
                    optimizer.zero_grad()
                    # backward pass
                    # E_habana.backward(retain_graph=True)
                    E_habana.backward()
                    # the autograd hooks for all_reduce is not triggered when the
                    # module is traced, so we explicitly reduce the grads
                    # This should be removed when lazy mode is enabled
                    if args.distributed and not args.use_lazy_eval:
                        # flatten and unflatten the tensors to reduce num of all reduce calls
                        flat, tensor_list = flatten_tensor(dlrm_habana.top_l.parameters())
                        with torch.no_grad():
                            torch.distributed.all_reduce(flat)
                        outputs = unflatten_tensor(flat, tensor_list)
                        updated_outputs = update_tensors(dlrm_habana.top_l.parameters(), outputs)

                        flat, tensor_list = flatten_tensor(dlrm_habana.bot_l.parameters())
                        with torch.no_grad():
                            torch.distributed.all_reduce(flat)
                        outputs = unflatten_tensor(flat, tensor_list)
                        updated_outputs = update_tensors(dlrm_habana.bot_l.parameters(), outputs)

                    # print('Net after backward')
                    # dlrm_habana.printParamsAndGrads()

                    # print('---'*17)
                    # printHooks(hookF,'hookF')
                    # print('---'*17)
                    # printHooks(hookB,'hookB')
                    # print('---'*17)

                    # debug prints (check gradient norm)
                    # for l in mlp.layers:
                    #     if hasattr(l, 'weight'):
                    #          print(l.weight.grad.norm().item())

                    # optimizer

                    optimizer.step()
                    printFnTrace('Default optimizer Done')
                    # optimizer_habana_1.step()

                    apply_optimizer_update()
                    printFnTrace('Custom optimizer done')

                    import time

                if args.mlperf_logging:
                    total_time += iteration_time
                else:
                    t2 = time_wrap(use_gpu)
                    total_time += t2 - t1
                total_accu += A
                total_loss += L_habana * mbs
                total_iter += 1
                total_samp += mbs

                should_print = ((j + 1) % args.print_freq == 0) or (j + 1 == nbatches)
                should_test = (
                    (args.test_freq > 0) and
                    (args.data_generation == "dataset") and
                    (((j + 1) % args.test_freq == 0) or (j + 1 == nbatches))
                )

                # print time, loss and accuracy
                if should_print or should_test:
                    gT = 1000.0 * total_time / total_iter if args.print_time else -1
                    total_time = 0

                    gA = total_accu / total_samp
                    total_accu = 0

                    gL = total_loss / total_samp
                    total_loss = 0

                    str_run_type = "inference" if args.inference_only else "training"
                    print(
                        "Finished {} it {}/{} of epoch {}, {:.2f} ms/it, ".format(
                            str_run_type, j + 1, nbatches, k, gT
                        ) +
                        "loss {:.6f}, accuracy {:3.3f} %".format(gL, gA * 100)
                    )
                    # Uncomment the line below to print out the total time with overhead
                    # print("Accumulated time so far: {}" \
                    # .format(time_wrap(use_gpu) - accum_time_begin))
                    total_iter = 0
                    total_samp = 0

                # testing
                if should_test and not args.inference_only:
                    # don't measure training iter time in a test iteration
                    if args.mlperf_logging:
                        previous_iteration_time = None

                    test_accu = 0
                    test_loss = 0
                    test_samp = 0

                    accum_test_time_begin = time_wrap(use_gpu)
                    if args.mlperf_logging:
                        scores = []
                        targets = []

                    for i, (X_test, lS_o_test, lS_i_test, T_test) in enumerate(test_ld):
                        # early exit if nbatches was set by the user and was exceeded
                        if (nbatches > 0 and i >= nbatches)  or X_test.size()[0] < args.test_mini_batch_size:
                            break

                        t1_test = time_wrap(use_gpu)

                        if args.distributed:
                            test_batch_size = int(X_test.size()[0]/args.world_size)
                            X_test = np.take(X_test, np.arange(args.rank * test_batch_size, (args.rank + 1) * test_batch_size),0)
                            if valid_device_emb_table is not None:
                                lS_o_test = np.take(lS_o_test, valid_device_emb_table, 0)
                                lS_i_test = itemgetter(*valid_device_emb_table)(lS_i_test)
                            else:
                                lS_o_test = []
                                lS_i_test = []
                            T_test = T_test[args.rank * test_batch_size : (args.rank + 1) * test_batch_size]
                            if isinstance(lS_i_test, tuple):
                                lS_i_test = list(lS_i_test)
                            else:
                                lS_i_test = [lS_i_test]

                        # embedding bag on Habana needs the last offset to be additionally appended which is pointing to end of indices
                        lS2_o_scratch = []
                        for kk in range(len(lS_i_test)):
                            lS2_o_scratch.append(torch.unsqueeze(
                                torch.cat((lS_o_test[kk], torch.tensor([lS_i_test[kk].numel()])), dim=0), dim=0))

                        lS_o_test = torch.cat(tuple(lS2_o_scratch), dim=0)
                        #print('lS_o after updating for last offset',lS_o)
                        #print("lS_i_test: ", lS_i_test)

                        with torch.no_grad():
                            for kk, sparse_index_group_batch in enumerate(lS_i_test):
                                sparse_offset_group_batch = lS_o_test[kk]
                                if not args.distributed:
                                    apply_preproc(sparse_offset_group_batch, sparse_index_group_batch, kk, m_spa, ln_emb)
                                else:
                                    apply_preproc(sparse_offset_group_batch, sparse_index_group_batch, kk, m_spa, ln_emb[valid_device_emb_table])

                        lS2_o_test = [S_o.to(torch.int32).to(device, non_blocking=True) for S_o in lS_o_test] if isinstance(lS_o_test, list) \
                            else lS_o.to(torch.int32).to(device, non_blocking=True)

                        X_test_device = X_test.to(device, non_blocking=True)

                        # forward pass
                        Z_test = dlrm_habana_wrap(model_to_run, X_test_device, lS2_o_test, gv.indices_fwd, gv.valid_count_fwd, gv.outputRowOffsets_hpu, gv.indices_bwd, gv.valid_count_bwd, gv.coalesced_grads)

                        if args.mlperf_logging:
                            S_test = Z_test.detach().cpu().numpy()  # numpy array
                            T_test = T_test.detach().cpu().numpy()  # numpy array
                            scores.append(S_test)
                            targets.append(T_test)
                        else:
                            # loss
                            E_test = loss_fn_wrap(Z_test, T_test, use_gpu, use_hpu, device)

                            # compute loss and accuracy
                            L_test = E_test.detach().cpu().numpy()  # numpy array
                            S_test = Z_test.detach().cpu().numpy()  # numpy array
                            T_test = T_test.detach().cpu().numpy()  # numpy array
                            mbs_test = T_test.shape[0]  # = mini_batch_size except last
                            A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))
                            test_accu += A_test
                            test_loss += L_test * mbs_test
                            test_samp += mbs_test

                        # print('Finish testing it ',i)
                        t2_test = time_wrap(use_gpu)

                    if args.mlperf_logging:
                        scores = np.concatenate(scores, axis=0)
                        targets = np.concatenate(targets, axis=0)

                        metrics = {
                            'loss': sklearn.metrics.log_loss,
                            'recall': lambda y_true, y_score:
                            sklearn.metrics.recall_score(
                                y_true=y_true,
                                y_pred=np.round(y_score)
                            ),
                            'precision': lambda y_true, y_score:
                            sklearn.metrics.precision_score(
                                y_true=y_true,
                                y_pred=np.round(y_score)
                            ),
                            'f1': lambda y_true, y_score:
                            sklearn.metrics.f1_score(
                                y_true=y_true,
                                y_pred=np.round(y_score)
                            ),
                            'ap': sklearn.metrics.average_precision_score,
                            'roc_auc': sklearn.metrics.roc_auc_score,
                            'accuracy': lambda y_true, y_score:
                            sklearn.metrics.accuracy_score(
                                y_true=y_true,
                                y_pred=np.round(y_score)
                            ),
                            # 'pre_curve' : sklearn.metrics.precision_recall_curve,
                            # 'roc_curve' :  sklearn.metrics.roc_curve,
                        }

                        # print("Compute time for validation metric : ", end="")
                        # first_it = True
                        validation_results = {}
                        for metric_name, metric_function in metrics.items():
                            # if first_it:
                            #     first_it = False
                            # else:
                            #     print(", ", end="")
                            # metric_compute_start = time_wrap(False)
                            validation_results[metric_name] = metric_function(
                                targets,
                                scores
                            )
                            # metric_compute_end = time_wrap(False)
                            # met_time = metric_compute_end - metric_compute_start
                            # print("{} {:.4f}".format(metric_name, 1000 * (met_time)),
                            #      end="")
                        # print(" ms")
                        gA_test = validation_results['accuracy']
                        gL_test = validation_results['loss']
                    else:
                        gA_test = test_accu / test_samp
                        gL_test = test_loss / test_samp

                    is_best = gA_test > best_gA_test
                    if is_best:
                        best_gA_test = gA_test
                        if not (args.save_model == ""):
                            print("Saving model to {}".format(args.save_model))
                            torch.save(
                                {
                                    "epoch": k,
                                    "nepochs": args.nepochs,
                                    "nbatches": nbatches,
                                    "nbatches_test": nbatches_test,
                                    "iter": j + 1,
                                    "state_dict": dlrm.state_dict(),
                                    "train_acc": gA,
                                    "train_loss": gL,
                                    "test_acc": gA_test,
                                    "test_loss": gL_test,
                                    "total_loss": total_loss,
                                    "total_accu": total_accu,
                                    "opt_state_dict": optimizer.state_dict(),
                                },
                                args.save_model,
                            )

                    if args.mlperf_logging:
                        is_best = validation_results['roc_auc'] > best_auc_test
                        if is_best:
                            best_auc_test = validation_results['roc_auc']

                        print(
                            "Testing at - {}/{} of epoch {},".format(j + 1, nbatches, k) +
                            " loss {:.6f}, recall {:.4f}, precision {:.4f},".format(
                                validation_results['loss'],
                                validation_results['recall'],
                                validation_results['precision']
                            ) +
                            " f1 {:.4f}, ap {:.4f},".format(
                                validation_results['f1'],
                                validation_results['ap'],
                            ) +
                            " auc {:.4f}, best auc {:.4f},".format(
                                validation_results['roc_auc'],
                                best_auc_test
                            ) +
                            " accuracy {:3.3f} %, best accuracy {:3.3f} %".format(
                                validation_results['accuracy'] * 100,
                                best_gA_test * 100
                            )
                        )
                    else:
                        print(
                            "Testing at - {}/{} of epoch {},".format(j + 1, nbatches, 0) +
                            " loss {:.6f}, accuracy {:3.3f} %, best {:3.3f} %".format(
                                gL_test, gA_test * 100, best_gA_test * 100
                            )
                        )
                    # Uncomment the line below to print out the total time with overhead
                    # print("Total test time for this group: {}" \
                    # .format(time_wrap(use_gpu) - accum_test_time_begin))

                    if (args.mlperf_logging and
                        (args.mlperf_acc_threshold > 0) and
                            (best_gA_test > args.mlperf_acc_threshold)):
                        print("MLPerf testing accuracy threshold " +
                              str(args.mlperf_acc_threshold) +
                              " reached, stop training")
                        break

                    if (args.mlperf_logging and
                        (args.mlperf_auc_threshold > 0) and
                            (best_auc_test > args.mlperf_auc_threshold)):
                        print("MLPerf testing auc threshold " +
                              str(args.mlperf_auc_threshold) +
                              " reached, stop training")
                        break

            k += 1  # nepochs

    # profiling
    if args.enable_profiling:
        with open("dlrm_s_pytorch.prof", "w") as prof_f:
            prof_f.write(prof.key_averages().table(sort_by="cpu_time_total"))
            prof.export_chrome_trace("./dlrm_s_pytorch.json")
        # print(prof.key_averages().table(sort_by="cpu_time_total"))

    # plot compute graph
    # print(dlrm_habana)
    if args.plot_compute_graph:
        # sys.exit(
        #     "ERROR: Please install pytorchviz package in order to use the"
        #     + " visualization. Then, uncomment its import above as well as"
        #     + " three lines below and run the code again."
        # )
        V = Z_habana.mean() if args.inference_only else E_habana
        dot = make_dot(V, params=dict(dlrm_habana.named_parameters()))
        dot.render('dlrm_s_pytorch_graph_hpu_custom_2')  # write .pdf file

    # print('Net at Finish')
    # dlrm_habana.printParamsAndGrads(grads=False)

    # test prints
    if not args.inference_only and args.debug_mode:
        print("updated parameters (weights and bias):")
        for param in dlrm_habana.parameters():
            print(param.detach().cpu().numpy())

    # export the model in onnx
    if args.save_onnx:
        with open("dlrm_s_pytorch.onnx", "w+b") as dlrm_pytorch_onnx_file:
            (X, lS_o, lS_i, _) = train_data[0]  # get first batch of elements
            torch.onnx._export(
                dlrm, (X, lS_o, lS_i), dlrm_pytorch_onnx_file, verbose=True
            )
        # recover the model back
        dlrm_pytorch_onnx = onnx.load("dlrm_s_pytorch.onnx")
        # check the onnx model
        onnx.checker.check_model(dlrm_pytorch_onnx)
