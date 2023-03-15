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
# Author: Sophie Henning

import torch
import torch.nn as nn
from collections import OrderedDict
from pathlib import Path
from typing import List, Dict, Optional

import modules.embeddings as embeddings
from .embedder import Embedder
from data_handling.sentence_with_modal_sense import SentenceWithModalSense
from util.util import prepend_optional_path


class TransformerBasedEmbedder(Embedder):
    def __init__(self, input_embeddings: embeddings.TransformerWrapper, embeddings_to_use: List[str],
                 additional_layers: Optional[Dict] = None):
        super(TransformerBasedEmbedder, self).__init__()
        self.embedding = input_embeddings
        self.embeddings_to_use = embeddings_to_use
        num_embeddings_to_use = len(self.embeddings_to_use)
        self.use_cls_embedding = "cls" in self.embeddings_to_use
        self.use_modal_embedding = "modal" in self.embeddings_to_use

        self.output_dim = self.embedding.embedding_dim

        self.has_additional_layer = additional_layers is not None
        if self.has_additional_layer:
            layer_args = additional_layers["args"]
            dropout = layer_args["dropout"]
            layer_type = getattr(nn, additional_layers["type"])
            num_layers = layer_args["num_layers"] if "num_layers" in layer_args.keys() else 1

            # Use nn.Sequential for variable number of additional layers
            # The first layer can have a different input dimension than the following ones
            # (depending on the number of token embeddings used), thus needs special treatment here
            input_dim = num_embeddings_to_use * self.output_dim
            module_dict = [('dropout1', nn.Dropout(p=dropout)),
                           ('layer1', layer_type(in_features=input_dim, out_features=self.output_dim)),
                           ('relu1', nn.ReLU())]
            for i in range(1, num_layers):
                module_dict += [(f'dropout{i+1}', nn.Dropout(p=dropout)),
                                (f'layer{i+1}', layer_type(in_features=self.output_dim, out_features=self.output_dim)),
                                (f'relu{i+1}', nn.ReLU())]
            self.additional_layer = nn.Sequential(OrderedDict(module_dict))
        else:
            self.output_dim = num_embeddings_to_use * self.embedding.embedding_dim

    @classmethod
    def from_args_dict(cls, args_dict: Dict, repo_path: Optional[Path]=None):
        embeddings_to_use = args_dict["embeddings_to_use"]
        use_cls_token = "cls" in embeddings_to_use
        if not (use_cls_token or "modal" in embeddings_to_use):
            raise NotImplementedError("Invalid configuration: Must use at least one of CLS or modal embedding")
        if "additional_layer" in args_dict.keys():
            additional_layer = args_dict["additional_layer"]
        else:
            additional_layer = None

        embedding_args = args_dict["embeddings"]
        transformer_args = embedding_args["args"]
        transformer_path = prepend_optional_path(Path(transformer_args["model_path"]), repo_path)
        layers_to_freeze = set(transformer_args["layers_to_freeze"]) if "layers_to_freeze" in transformer_args else None
        transformer = \
            getattr(embeddings, embedding_args["type"])(transformer_path,
                                                        fine_tune=transformer_args["fine_tune"],
                                                        token_mask_prob=transformer_args["token_mask_prob"],
                                                        output_dropout=transformer_args["dropout"],
                                                        keep_cls_token=use_cls_token,
                                                        layers_to_freeze=layers_to_freeze)

        return cls(transformer, embeddings_to_use, additional_layer)

    def forward(self, sentence_batch: List[SentenceWithModalSense], device: torch.device) -> torch.Tensor:
        """
        Perform a forward pass through the network.
        Args:
           sentence_batch: List of lists of tokens (one sub-list: one sentence)
           device: the device the input shall be sent to

        Returns:
            logits: output logits with shape(batch_size, n_classes)
        """
        list_of_tokens_lists = [sentence.tokens for sentence in sentence_batch]
        # Get the embedding tensor for the inputs. Output shape: (batch_size, max_seq_length, embed_dim)
        # TransformerWrapper returns a tuple (embeddings, true_sequence_lengths)
        x_embedded = self.embedding(list_of_tokens_lists)[0]

        if self.use_cls_embedding:
            # Output shape: (batch_size, embed_dim)
            x_cls = x_embedded[:, 0, :]
            if self.use_modal_embedding:
                x_modal = self._get_modal_verb_embedding(sentence_batch, x_embedded)
                x_lin = torch.cat((x_cls, x_modal), 1)
            else:
                x_lin = x_cls
        else:
            # Use embedding of modal verb only
            x_lin = self._get_modal_verb_embedding(sentence_batch, x_embedded)

        if self.has_additional_layer:
            return self.additional_layer(x_lin)

        return x_lin

    def _get_modal_verb_embedding(self, sentence_batch: List[SentenceWithModalSense], x_embedded: torch.Tensor) -> torch.Tensor:
        # If we don't use the CLS embedding, it is not stored in x_embedded and the n-th token in the sentence
        # (counting from 0) can be retrieved from x_embedded via [:, n, :]
        # If we use the CLS embedding, we need to add 1 to the position index to account for its presence in x_embedde
        increment = 1 if self.use_cls_embedding else 0
        # However, the position of the modal is not constant

        # Store the positions of the modal verbs in a list
        positions = torch.tensor([sentence.target_modal_position + increment for sentence in sentence_batch],
                                 device=x_embedded.device)

        # From matrix i, take row positions[i]
        return x_embedded[torch.arange(len(sentence_batch), device=x_embedded.device), positions, :]