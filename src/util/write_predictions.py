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
from pathlib import Path
from typing import Dict, OrderedDict, Optional, List

from data_handling.tasks import NO_DOMAINS, ModalTask
from data_handling.sentence_with_modal_sense import SentenceWithModalSense


def write_header(task2corpus_path: Dict[str, Path], tasks: OrderedDict[str, ModalTask], output_paths: Dict[str, Path]) -> None:
    """Write header in each task's prediction file"""
    for task_name, corpus_path in task2corpus_path.items():
        task = tasks[task_name]
        with open(output_paths[task_name], "w", encoding="utf-8") as out:
            out.write(f"Evaluation on {corpus_path}\n")
            if task.target_domains != NO_DOMAINS:
                out.write("Domain\t")
            if task.has_documents:
                out.write("Document\t")
            plural = "" if task.single_label_task else "s"
            vocab = task.output_vocab
            vocab_str = ",".join([f"{ix}:{vocab.ix2token(ix)}" for ix in range(len(vocab))])
            out.write(
                f"Sentence ID\tTarget modal\tTarget modal position\tTarget label{plural}\tPredicted label{plural}"
                f"\tSentence text\tPrediction confidences (vocab: {vocab_str})\n")


def write_predictions_to_output_files(sentences: List[SentenceWithModalSense],
                                      task2domain2modal2preds: Dict[str, Dict[str, Dict[str, Optional[torch.Tensor]]]],
                                      task2domain2modal2tgts: Dict[str, Dict[str, Dict[str, Optional[torch.Tensor]]]],
                                      task2domain2modal2confs: Dict[str, Dict[str, Dict[str, Optional[torch.Tensor]]]],
                                      task2domain2modal2mask: Dict[str, Dict[str, Dict[str, List[int]]]],
                                      tasks: OrderedDict[str, ModalTask], output_paths: Dict[str, Path]) -> None:
    """Write given predictions to per-task prediction files"""
    for task_name, task in tasks.items():
        domains = task.target_domains
        modals = task.target_modals
        output_vocab = task.output_vocab

        with open(output_paths[task_name], "a", encoding="utf-8", errors='replace') as out:
            for domain in domains:
                for modal in modals:
                    tgts = task2domain2modal2tgts[task_name][domain][modal]
                    preds = task2domain2modal2preds[task_name][domain][modal]
                    confs = task2domain2modal2confs[task_name][domain][modal]

                    # Retrieve the correct sentence(s) per domain-modal combination using the masks that
                    # were used to select the correct predictions/targets/confidences
                    mask = task2domain2modal2mask[task_name][domain][modal]
                    # Masks are list of indices in the batch
                    # -> Index w.r.t. this list is also a valid index in preds/tgts/confs
                    for submatrix_ix in range(len(mask)):
                        # Batch index stored in mask
                        batch_ix = mask[submatrix_ix]
                        sentence = sentences[batch_ix]
                        pred = preds[submatrix_ix]
                        tgt = tgts[submatrix_ix]
                        conf = confs[submatrix_ix]

                        # Convert prediction and targets into human-readable labels
                        if task.single_label_task:
                            pred_string = output_vocab.ix2token(pred.item())
                            target_string = output_vocab.ix2token(tgt.item())
                        else:
                            # Multi-labeling
                            pred_lbls = []
                            tgt_lbls = []
                            for j in range(len(output_vocab)):
                                label = output_vocab.ix2token(j)
                                if pred[j] == 1:
                                    pred_lbls.append(label)
                                if tgt[j] == 1:
                                    tgt_lbls.append(label)
                            pred_string = "-".join(pred_lbls)
                            target_string = "-".join(tgt_lbls)

                        ln = f"{sentence.domain}\t" if task.target_domains != NO_DOMAINS else ""
                        ln = f"{ln}{sentence.doc}\t" if task.has_documents else ln
                        ln = f"{ln}{sentence.id}\t{sentence.target_modal}" \
                            f"\t{sentence.target_modal_position}\t{target_string}\t{pred_string}" \
                            f"\t{' '.join(sentence.tokens)}\t{str(conf)}\n"

                        out.write(ln)
