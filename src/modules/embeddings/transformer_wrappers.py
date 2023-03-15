# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Authors: Stefan Grünewald,Sophie Henning

import torch
import random

from torch import nn

from transformers import BertTokenizer
from transformers import BertModel, BertConfig

from transformers import RobertaTokenizer
from transformers import RobertaModel, RobertaConfig

from transformers import XLMRobertaTokenizer
from transformers import XLMRobertaModel, XLMRobertaConfig

from transformers import ElectraTokenizer
from transformers import ElectraModel, ElectraConfig

from transformers import AlbertTokenizer
from transformers import AlbertModel, AlbertConfig

from transformers import AutoTokenizer, AutoModel, AutoConfig

from torch.nn import Dropout

from .scalar_mix import ScalarMixWithDropout

from typing import Optional, Set


class TransformerWrapper(nn.Module):
    """Base class for turning batches of sentences into tensors of (BERT/RoBERTa/...) embeddings.

    An object of this class takes as input a bunches of sentences (represented as a lists of lists of tokens) and
    returns, for each specified output ID, tensors (shape: batch_size * max_sent_len * embedding_dim) of token
    embeddings. The embeddings for the different outputs are generated using the same underlying transformer model, but
    by default use different scalar mixtures of the internal transformer layers to generate final embeddings.
    """

    def __init__(self, model_class, tokenizer_class, config_class, model_path, output_ids=None, tokenizer_path=None,
                 config_only=False, fine_tune=True, shared_embeddings=None, hidden_dropout=0.1, attn_dropout=0.1,
                 output_dropout=0.5, scalar_mix_layer_dropout=0.1, token_mask_prob=0.2, word_piece_pooling="first",
                 keep_cls_token: bool = False, layers_to_freeze: Optional[Set[int]] = None):
        """
        Args:
            model_class: Class of transformer model to use for token embeddings.
            tokenizer_class: Class of tokenizer to use for tokenization.
            config_class: Class of transformer config.
            model_path: Path to load transformer model from.
            output_ids: List of output IDs to generate embeddings for. These outputs will get separately trained
              scalar mixtures. If none provided, there will be only one scalar mix and one output.
            tokenizer_path: Path to load tokenizer from (default: None; specify when using config_only option).
            config_only: If True, only load model config, not weights (default: False).
            fine_tune: Whether to fine-tune the transformer language model. If False, weights of the transformer model
              will not be trained. Default: True.
            shared_embeddings: If specified (as list of lists of output IDs), the specified groups of outputs will
              share the same scalar mixture (and thus embeddings). Default: None.
            hidden_dropout: Dropout ratio for hidden layers of the transformer model.
            attn_dropout: Dropout ratio for the attention probabilities.
            output_dropout: Dropout ratio for embeddings output.
            scalar_mix_layer_dropout: Dropout ratio for transformer layers.
            token_mask_prob: Probability of replacing input tokens with mask token.
            word_piece_pooling: How to combine multiple word piece embeddings into one token embedding. Default: "first".
            keep_cls_token: Set to True if the CLS special token should be kept so that later on its embedding can be
            extracted.
            layers_to_freeze: Set this to a set of integer indices if you want to freeze some of the transformer layer
             (n will be interpreted as the n-th layer, 0 is the embedding layer).
        """
        super(TransformerWrapper, self).__init__()

        if not output_ids:
            self.output_ids = ["__dummy_output__"]
        else:
            self.output_ids = output_ids

        self.layers_to_freeze = layers_to_freeze
        self.model, self.tokenizer = self._init_model(model_class, tokenizer_class, config_class,
                                                      model_path, tokenizer_path, config_only=config_only,
                                                      hidden_dropout=hidden_dropout, attn_dropout=attn_dropout)

        self.token_mask_prob = token_mask_prob
        self.embedding_dim = self.model.config.hidden_size
        self.fine_tune = fine_tune

        self.scalar_mix = self._init_scalar_mix(shared_embeddings=shared_embeddings,
                                                layer_dropout=scalar_mix_layer_dropout)
        self.word_piece_pooling = word_piece_pooling
        self.keep_cls_token = keep_cls_token

        if output_dropout > 0.0:
            self.output_dropout = Dropout(p=output_dropout)

    @classmethod
    def from_args_dict(cls, args_dict, model_dir=None):
        if model_dir is not None:
            args_dict["config_only"] = True
            args_dict["model_path"] = model_dir
            args_dict["tokenizer_path"] = model_dir / "tokenizer"

        return cls(**args_dict)

    def save_config(self, save_dir, prefix=""):
        """Save this module's transformer configuration to the specified directory."""
        (save_dir / prefix).mkdir(parents=True, exist_ok=True)

        self.model.config.to_json_file(save_dir / prefix / "config.json")
        self.tokenizer.save_pretrained(save_dir / prefix / "tokenizer")

    def _init_model(self, model_class, tokenizer_class, config_class, model_path, tokenizer_path, config_only=False,
                    hidden_dropout=0.1, attn_dropout=0.1):
        """Initialize the transformer language model."""
        if config_only:
            model = model_class(config_class.from_json_file(str(model_path / "config.json")))
            tokenizer = tokenizer_class.from_pretrained(str(tokenizer_path))
        else:
            model = model_class.from_pretrained(model_path,
                                                output_hidden_states=True,
                                                hidden_dropout_prob=hidden_dropout,
                                                attention_probs_dropout_prob=attn_dropout,
                                                return_dict=True)
            tokenizer = tokenizer_class.from_pretrained(model_path)

        if self.layers_to_freeze is not None:
            # Want to change the set without removing the information about the frozen layers from the Wrapper
            indices_copy = self.layers_to_freeze.copy()

            # Check for embedding layer
            layers_to_freeze = [model.embeddings] if 0 in indices_copy else []  # 0: embedding layer

            # Remove 0 from set copy (not a valid index for the encoder layers)
            indices_copy.discard(0)

            # Get the encoder layers
            for layer_index in indices_copy:
                layers_to_freeze.append(model.encoder.layer[layer_index-1])  # -1 because n-th layer is (n-1)-th element in encoder layer list

            for layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

        return model, tokenizer

    def _init_scalar_mix(self, shared_embeddings=None, layer_dropout=0.1):
        """Initialize the scalar mixture module."""
        num_layers = self.model.config.num_hidden_layers + 1  # Add 1 because of input embeddings

        if shared_embeddings is None:
            scalar_mix = nn.ModuleDict(
                {output_id: ScalarMixWithDropout(mixture_size=num_layers, layer_dropout=layer_dropout) for
                 output_id in self.output_ids})
        else:
            scalar_mix = nn.ModuleDict()
            for group in shared_embeddings:
                curr_scalarmix = ScalarMixWithDropout(mixture_size=num_layers, layer_dropout=layer_dropout)
                for outp_id in group:
                    scalar_mix[outp_id] = curr_scalarmix
            for outp_id in self.output_ids:
                if outp_id not in scalar_mix:
                    # Add scalar mixes for all outputs that don't have one yet
                    scalar_mix[outp_id] = ScalarMixWithDropout(mixture_size=num_layers, layer_dropout=layer_dropout)

        return scalar_mix

    def forward(self, input_sentences):
        """Transform a bunch of input sentences (list of lists of tokens) into a batch (tensor) of
        BERT/RoBERTa/etc. embeddings.

        Args:
            input_sentences: The input sentences to transform into embeddings (list of lists of tokens).

        Returns: A tuple consisting of (a) a dictionary with the embeddings for each output/annotation ID
          (shape: batch_size * max_seq_len * embedding_dim) or a tensor with the embeddings if there is only one output
           structure; (b) a tensor containing the length (number of tokens) of each sentence (shape: batch_size).
        """
        # Retrieve inputs for BERT model
        tokens, token_lengths, word_piece_ids, attention_mask = self._get_model_inputs(input_sentences)

        # Get embeddings tensors (= a dict containing one tensor for each output)
        raw_embeddings = self._get_raw_embeddings(word_piece_ids, attention_mask)

        # For each output, extract the token embeddings
        processed_embeddings = dict()
        for output_id in self.output_ids:
            processed_embeddings[output_id] = self._process_embeddings(raw_embeddings[output_id], tokens, token_lengths)

        # Sav true sequence lengths in a tensor
        true_seq_lengths = self._compute_true_seq_lengths(input_sentences, device=tokens.device)

        # If there is only one output, get rid of the dummy output ID
        if processed_embeddings.keys() == {"__dummy_output__"}:
            processed_embeddings = processed_embeddings["__dummy_output__"]

        return processed_embeddings, true_seq_lengths

    def _get_model_inputs(self, input_sentences):
        """Take a list of sentences and return tensors for token IDs, attention mask, and original token mask"""
        mask_prob = self.token_mask_prob if self.training else 0.0
        input_sequences = [TransformerInputSequence(sent, self.tokenizer, token_mask_prob=mask_prob,
                                                    treat_cls_token_as_regular_token=self.keep_cls_token)
                           for sent in input_sentences]
        device = next(iter(self.model.parameters())).device  # Ugly :(

        return TransformerInputSequence.batchify(input_sequences, device)

    def _get_raw_embeddings(self, word_piece_ids, attention_mask):
        """Take tensors for input tokens and run them through underlying BERT-based model, performing the learned scalar
         mixture for each output"""
        raw_embeddings = dict()

        with torch.set_grad_enabled(self.fine_tune):
            # Checked in debugger: this doesn't override layer-wise freezing
            embedding_layers = torch.stack(self.model(word_piece_ids, attention_mask=attention_mask).hidden_states)

        for output_id in self.output_ids:
            if self.output_dropout:
                embedding_layers_with_dropout = self.output_dropout(embedding_layers)
                curr_output = self.scalar_mix[output_id](embedding_layers_with_dropout)
            else:
                curr_output = self.scalar_mix[output_id](embedding_layers)
            raw_embeddings[output_id] = curr_output

        return raw_embeddings

    def _process_embeddings(self, raw_embeddings, tokens, token_lengths):
        """Pool the raw word piece embeddings into token embeddings using the specified method."""
        batch_size = raw_embeddings.shape[0]
        embeddings_dim = raw_embeddings.shape[2]

        # Attach "neutral element" / padding to the raw embeddings tensor
        neutral_element = -1e10 if self.word_piece_pooling == "max" else 0.0  # Negative "infinity" for max pooling
        neutral_element_t = torch.empty((1, 1, embeddings_dim), dtype=torch.float, device=raw_embeddings.device).fill_(neutral_element)
        neutral_element_exp = neutral_element_t.expand((batch_size, 1, embeddings_dim))
        embeddings_with_neutral = torch.cat((raw_embeddings, neutral_element_exp), dim=1)

        # Gather the word piece embeddings corresponding to each token
        assert tokens.shape[0] == batch_size
        max_num_tokens = tokens.shape[1]
        max_wp_in_token = tokens.shape[2]

        # # NumPy advanced indexing version to get embeddings
        # indexing_start = time.time()
        # tokens1 = tokens.view(batch_size, max_num_tokens * max_wp_in_token)
        # # indexed_embeddings: Shape (batch_size, batch_size, max_num_tokens * max_wp_in_token, embeddings_dim)
        # indexed_embeddings = embeddings_with_neutral[:, tokens1]
        # # indexed_embeddings[i, i, :, :, :] == gathered_embeddings[i,:,:]
        # # -> Select these matrices and turn into (batch_size, max_num_tokens*max_wp_in_token, embeddings_dim)
        # batch_size_range = torch.arange(batch_size, device=raw_embeddings.device)
        # indexed_embeddings = indexed_embeddings[batch_size_range, batch_size_range, :]
        # indexed_embeddings = indexed_embeddings.view(batch_size, max_num_tokens, max_wp_in_token, embeddings_dim)
        # print(f"Process embeddings: advanced indexing version took {time.time()-indexing_start:.5f} seconds")
        #
        # # torch.gather version
        # gather_start = time.time()
        tokens = tokens.view(batch_size, max_num_tokens * max_wp_in_token).unsqueeze(-1).expand(
            (-1, -1, embeddings_dim))
        gathered_embeddings = torch.gather(embeddings_with_neutral, 1, tokens).view(batch_size, max_num_tokens,
                                                                                    max_wp_in_token, embeddings_dim)
        # print(f"Process embeddings: gather version took {time.time() - gather_start:.5f} seconds")
        #
        # assert torch.all(torch.eq(gathered_embeddings, indexed_embeddings))

        # Pool values using the specified method
        if self.word_piece_pooling == "first":
            token_embeddings = gathered_embeddings[:, :, 0, :]
        elif self.word_piece_pooling == "sum":
            token_embeddings = torch.sum(gathered_embeddings, dim=2)
        elif self.word_piece_pooling == "avg":
            token_embeddings = torch.sum(gathered_embeddings, dim=2) / token_lengths.unsqueeze(-1)
        elif self.word_piece_pooling == "max":
            token_embeddings, _ = torch.max(gathered_embeddings, dim=2)
        else:
            raise Exception(f"Unknown pooling method \"{self.word_piece_pooling}\"!")

        return token_embeddings

    def _compute_true_seq_lengths(self, sentences, device=None):
        return torch.tensor([len(sent) for sent in sentences], device=device)

    def parallelize(self, device_ids):
        """Parallelize this module for multi-GPU setup-"""
        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)


class BertWrapper(TransformerWrapper):
    """Embeddings wrapper class for modules based on BERT."""

    def __init__(self, *args, **kwargs):
        super(BertWrapper, self).__init__(BertModel, BertTokenizer, BertConfig, *args, **kwargs)


class RobertaWrapper(TransformerWrapper):
    """Embeddings wrapper class for modules based on RoBERTa."""

    def __init__(self, *args, **kwargs):
        super(RobertaWrapper, self).__init__(RobertaModel, RobertaTokenizer, RobertaConfig, *args, **kwargs)


class XLMRobertaWrapper(TransformerWrapper):
    """Embeddings wrapper class for modules based on XLM-R."""

    def __init__(self, *args, **kwargs):
        super(XLMRobertaWrapper, self).__init__(XLMRobertaModel, XLMRobertaTokenizer, XLMRobertaConfig, *args, **kwargs)


class ElectraWrapper(TransformerWrapper):
    """Embeddings wrapper class for modules based on ELECTRA."""

    def __init__(self, *args, **kwargs):
        super(ElectraWrapper, self).__init__(ElectraModel, ElectraTokenizer, ElectraConfig, *args, **kwargs)


class AlbertWrapper(TransformerWrapper):
    """Embeddings wrapper class for modules based on ELECTRA."""

    def __init__(self, *args, **kwargs):
        super(AlbertWrapper, self).__init__(AlbertModel, AlbertTokenizer, AlbertConfig, *args, **kwargs)


class OtherWrapper(TransformerWrapper):
    """Embeddings wrapper class for any pre-trained module"""
    def __init__(self, model_path, *args, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)

        super(OtherWrapper, self).__init__(model, tokenizer, config, model_path=model_path, *args, **kwargs)


class TransformerInputSequence:
    """Class for representing the features of a single, dependency-annotated sentence in tensor
    form, for usage in transformer-based models such as BERT.

    Example (BERT):
    ```
    Input sentence:                 Beware      the     jabberwock                 ,    my   son    !
    BERT word pieces:      [CLS]    be ##ware   the     ja ##bber   ##wo  ##ck     ,    my   son    ! [SEP]  ([PAD] [PAD] [PAD]  ...)
    BERT word piece IDs:     101  2022   8059  1996  14855  29325  12155  3600  1010  2026  2365  999   102  (    0     0     0  ...)
    BERT attention mask:       1     1      1     1      1      1      1     1     1     1     1    1     1  (    0     0     0  ...)

    Word-to-pieces mapping (non-padded): [[1,2], [3], [4,5,6,7], [8], [9], [10], [11]]
    ```
    Word-to-pieces mapping is built & stored in self.tokens!
    """

    def __init__(self, orig_tokens, tokenizer, token_mask_prob=0.0, treat_cls_token_as_regular_token: bool = False):
        """
        Args:
            orig_tokens: Tokens to convert into a BertInputSequence.
            tokenizer: Tokenizer to use to split original tokens into word pieces.
            token_mask_prob: Probability of replacing an input token with a mask token. All word pieces of a given token
              will be replaced.
            treat_cls_token_as_regular_token: set to True if the CLS token should treated as a regular token
            (be retrievable through the word-to-pieces mapping)
        """
        self.tokenizer = tokenizer

        self.word_pieces = list()
        self.attention_mask = list()
        self.tokens = list()
        self.token_lengths = list()

        cls_index = 0 if treat_cls_token_as_regular_token else None
        self.append_special_token(self.tokenizer.cls_token, cls_index)  # BOS marker

        ix = 1
        for orig_token in orig_tokens:
            tok_length = self.append_regular_token(orig_token, ix, mask_prob=token_mask_prob)
            ix += tok_length

        self.append_special_token(self.tokenizer.sep_token)  # EOS marker

        sequence_length_to_expect = len(orig_tokens) + 1 if treat_cls_token_as_regular_token else len(orig_tokens)
        assert sequence_length_to_expect == len(self.tokens) <= len(self.word_pieces) == len(self.attention_mask)

        # Convert word pieces to IDs
        self.word_piece_ids = self.tokenizer.convert_tokens_to_ids(self.word_pieces)

    def __len__(self):
        return len(self.word_pieces)

    def append_special_token(self, token, ix_for_treatment_as_regular_token: Optional[int] = None) -> None:
        """Append a special token (e.g. BOS token, MASK token) to the sequence. The token will receive attention in the
        model, but will not be counted as an original token.

        Args:
            ix_for_treatment_as_regular_token: Optional, set this to the index of the special token in the sequence if
            you want to later extract an embedding for this special token.
        """
        self.word_pieces.append(token)
        self.attention_mask.append(1)

        if ix_for_treatment_as_regular_token is not None:
            # Special token should be treated as a regular token
            # (= an embedding for this special token can later be extracted)
            # Word piece length is always 1 for special tokens
            word_piece_length = 1
            ix = ix_for_treatment_as_regular_token
            self.tokens.append(list(range(ix, ix + word_piece_length)))
            self.token_lengths.append(word_piece_length)

    def append_regular_token(self, token, ix, mask_prob=0.0):
        """Append regular token (i.e., a word from the input sentence) to the sequence. The token will be split further
        into word pieces by the tokenizer."""
        if isinstance(self.tokenizer, RobertaTokenizer):
            curr_word_pieces = self.tokenizer.tokenize(token, add_prefix_space=True)
        else:
            curr_word_pieces = self.tokenizer.tokenize(token)

        if len(curr_word_pieces) == 0:
            print("WARNING: Replacing non-existent token with UNK")
            curr_word_pieces = [self.tokenizer.unk_token]

        if mask_prob > 0.0 and random.random() < mask_prob:
            curr_word_pieces = [self.tokenizer.mask_token] * len(curr_word_pieces)

        self.word_pieces += curr_word_pieces
        self.attention_mask += [1] * len(curr_word_pieces)
        self.tokens.append(list(range(ix, ix + len(curr_word_pieces))))
        self.token_lengths.append(len(curr_word_pieces))

        return len(curr_word_pieces)

    def pad_to_length(self, padded_num_tokens, padded_num_word_pieces, padded_max_wp_per_token):
        """Pad the sentence to the specified length. This will increase the length of all fields to padded_length by
        adding the padding label/index."""
        wp_padding_length = padded_num_word_pieces - len(self.word_pieces)
        seq_padding_length = padded_num_tokens - len(self.tokens)

        assert wp_padding_length >= 0
        assert seq_padding_length >= 0
        assert padded_max_wp_per_token >= max(len(token) for token in self.tokens)

        self.word_pieces += [self.tokenizer.pad_token] * wp_padding_length
        self.word_piece_ids += [self.tokenizer.pad_token_id] * wp_padding_length
        self.attention_mask += [0] * wp_padding_length

        wp_padding_ix = padded_num_word_pieces
        for i, curr_token in enumerate(self.tokens):
            curr_padding_length = padded_max_wp_per_token - len(curr_token)
            self.tokens[i] = curr_token + [padded_num_word_pieces] * curr_padding_length

        self.tokens += [[padded_num_word_pieces] * padded_max_wp_per_token for _ in range(seq_padding_length)]
        self.token_lengths += [1] * seq_padding_length

        assert len(self.word_pieces) == len(self.word_piece_ids) == len(self.attention_mask)
        assert all(len(tok) == len(self.tokens[0]) for tok in self.tokens)

    def tensorize(self, device, padded_num_tokens=None, padded_num_word_pieces=None, padded_max_wp_per_token=None):
        if len(self.word_piece_ids) > 512:
            self._throw_out_non_first_word_pieces()
            assert len(self.word_piece_ids) <= 512

        if padded_num_tokens is None:
            padded_num_tokens = len(self.tokens)
        if padded_num_word_pieces is None:
            padded_num_word_pieces = len(self.word_pieces)
        if padded_max_wp_per_token is None:
            padded_max_wp_per_token = max(len(token) for token in self.tokens)

        self.pad_to_length(padded_num_tokens, padded_num_word_pieces, padded_max_wp_per_token)

        self.word_piece_ids = torch.tensor(self.word_piece_ids, device=device)
        self.attention_mask = torch.tensor(self.attention_mask, device=device)
        self.tokens = torch.tensor(self.tokens, device=device)
        self.token_lengths = torch.tensor(self.token_lengths, device=device)

    def _throw_out_non_first_word_pieces(self):
        self.word_pieces = [self.tokenizer.cls_token] + [self.word_pieces[tok[0]] for tok in self.tokens] + [
            self.tokenizer.sep_token]
        self.word_piece_ids = [self.tokenizer.cls_token_id] + [self.word_piece_ids[tok[0]] for tok in self.tokens] + [
            self.tokenizer.sep_token_id]
        self.attention_mask = [1] * (len(self.tokens) + 2)

        self.tokens = [[i + 1] for i in range(len(self.tokens))]
        self.token_lengths = [1] * len(self.tokens)

    @staticmethod
    def batchify(input_seqs, device):
        padded_num_tokens = max(len(input_seq.tokens) for input_seq in input_seqs)
        # Restrict maximum number of word pieces to 512
        max_wp_length = max(len(input_seq.word_pieces) for input_seq in input_seqs)
        while max_wp_length > 512:
            max_sequence = max(input_seqs, key=lambda item: len(item.word_pieces))
            max_sequence._throw_out_non_first_word_pieces()
            max_wp_length = max(len(input_seq.word_pieces) for input_seq in input_seqs)
        padded_num_word_pieces = max(len(input_seq.word_pieces) for input_seq in input_seqs)
        padded_max_wp_per_token = max(len(token) for input_seq in input_seqs for token in input_seq.tokens)

        for input_seq in input_seqs:
            input_seq.tensorize(device, padded_num_tokens=padded_num_tokens,
                                padded_num_word_pieces=padded_num_word_pieces,
                                padded_max_wp_per_token=padded_max_wp_per_token)

        tokens = torch.stack([input_seq.tokens for input_seq in input_seqs])
        token_lengths = torch.stack([input_seq.token_lengths for input_seq in input_seqs])
        word_piece_ids = torch.stack([input_seq.word_piece_ids for input_seq in input_seqs])
        attention_mask = torch.stack([input_seq.attention_mask for input_seq in input_seqs])

        return tokens, token_lengths, word_piece_ids, attention_mask