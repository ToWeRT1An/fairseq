# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from fairseq.models import FairseqDecoder
import torch
from fairseq import utils
class GroupIncrementalDecoder(FairseqDecoder):
    """Base class for incremental decoders.

    Incremental decoding is a special mode at inference time where the Model
    only receives a single timestep of input corresponding to the previous
    output token (for input feeding) and must produce the next output
    *incrementally*. Thus the model must cache any long-term state that is
    needed about the sequence, e.g., hidden states, convolutional states, etc.

    Compared to the standard :class:`FairseqDecoder` interface, the incremental
    decoder interface allows :func:`forward` functions to take an extra keyword
    argument (*incremental_state*) that can be used to cache state across
    time-steps.

    The :class:`FairseqIncrementalDecoder` interface also defines the
    :func:`reorder_incremental_state` method, which is used during beam search
    to select and reorder the incremental state based on the selection of beams.

    To learn more about how incremental decoding works, refer to `this blog
    <http://www.telesens.co/2019/04/21/understanding-incremental-decoding-in-fairseq/>`_.
    """

    def __init__(self, dictionary):
        super().__init__(dictionary)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        """
        Args:
            prev_output_tokens (LongTensor): shifted output tokens of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (dict, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict, optional): dictionary used for storing
                state during :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        raise NotImplementedError

    def extract_features(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        """
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        raise NotImplementedError

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder incremental state.

        This should be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """
        seen = set()

        def apply_reorder_incremental_state(module):
            if module != self and hasattr(module, 'reorder_incremental_state') \
                    and module not in seen:
                seen.add(module)
                module.reorder_incremental_state(incremental_state, new_order)

        self.apply(apply_reorder_incremental_state)

    def set_beam_size(self, beam_size):
        """Sets the beam size in the decoder and all children."""
        if getattr(self, '_beam_size', -1) != beam_size:
            seen = set()

            def apply_set_beam_size(module):
                if module != self and hasattr(module, 'set_beam_size') \
                        and module not in seen:
                    seen.add(module)
                    module.set_beam_size(beam_size)

            self.apply(apply_set_beam_size)
            self._beam_size = beam_size
    def get_normalized_probs(self, net_output, log_probs, sample):
        """Get normalized probabilities (or log probs) from a net's output."""

        def get_len_label(attns):
            
            labels = torch.zeros(attns.shape[0],attns.shape[-1]).to(attns.device)
            for i in range(attns.shape[0]):
                attn = attns[i]
                values, indices = torch.topk(attn,2)
                wrong_lines =(indices[:,0]==attn.shape[-1]-1)
                for j in range(len(wrong_lines)):
                    if wrong_lines[j]==1:
                        indices[j][0]=indices[j][-1]
                indices[len(wrong_lines)-1,0]=attn.shape[-1]-1

                label = torch.zeros(attn.shape).to(attns.device)
                label = label.scatter_(1,[indices[:,0]],1).sum(dim=0)
                labels[i]=label
            print('---------labels------')
            print(labels)
            return labels

        len_pre_labels = get_len_label(net_output[1]['attn']).long().view(-1,1)

        if hasattr(self, 'adaptive_softmax') and self.adaptive_softmax is not None:
            if sample is not None:
                assert 'target' in sample
                target = sample['target']
            else:
                target = None
            out1 = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            out2 = self.adaptive_softmax.get_log_prob(net_output[1]['len_pre'],target=len_pre_labels)
            return out1.exp_() if not log_probs else out1, out2.exp_() if not log_probs else out2,len_pre_labels

        logits = net_output[0]
        if log_probs:
            out1 = utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
            out2 = utils.log_softmax(net_output[1]['len_pre'],dim=-1,onnx_trace=self.onnx_trace)
            return out1,out2,len_pre_labels
        else:
            out1 = utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
            out2 = utils.softmax(net_output[1]['len_pre'],dim=-1,onnx_trace=self.onnx_trace)            
            return out1,out2,len_pre_labels
