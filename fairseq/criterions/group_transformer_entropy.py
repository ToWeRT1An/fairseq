# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

from fairseq import utils

from . import FairseqCriterion, register_criterion
from torchvision.utils import save_image
import time

@register_criterion('group_transformer_entropy')
class GroupTransformerEntropy(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])

        loss, nll_loss,acc = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'acc':acc
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs1,lprobs2,target2 = model.get_normalized_probs(net_output, log_probs=True)
        lprobs1 = lprobs1.view(-1, lprobs1.size(-1))
        lprobs2 = lprobs2.view(-1,lprobs2.size(-1))
        target1 = model.get_targets(sample, net_output).view(-1, 1)

        print('------lprobs2---')
        print(lprobs2.shape)
        print(lprobs2)
        print('------target2----')
        print(target2.shape)
        print(target2)
        
        non_pad_mask = target1.ne(self.padding_idx)
        nll_loss1 = -lprobs1.gather(dim=-1, index=target1)[non_pad_mask]
        smooth_loss1 = -lprobs1.sum(dim=-1, keepdim=True)[non_pad_mask]
        nll_loss2 = -lprobs2.gather(dim=-1, index=target2)
        smooth_loss2 = -lprobs2.sum(dim=-1,keepdim=True)

        if reduce:
            nll_loss1 = nll_loss1.sum()
            smooth_loss1 = smooth_loss1.sum()
            nll_loss2 = nll_loss2.sum()
            smooth_loss2 = smooth_loss2.sum()

        eps_i1 = self.eps / lprobs1.size(-1)
        loss1 = (1. - self.eps) * nll_loss1 + eps_i1 * smooth_loss1

        eps_i2 = self.eps / lprobs2.size(-1)
        loss2 = (1. - self.eps) * nll_loss2 + eps_i2 * smooth_loss2

        len_pre = torch.topk(lprobs2,1).squeeze(-1)
        acc1 = torch.eq(len_pre,target2).sum()/(len_pre.shape[0]*len_pre.shape[1])
        acc2 = torch.eq(len_pre.sum(dim=-1),target2.sum(dim=-1)).sum()/(len_pre.shape[0])

        return loss1, nll_loss1, acc2

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
