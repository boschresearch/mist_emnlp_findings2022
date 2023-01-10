# MiST data
``instance-wise`` contains the annotated MiST corpus, with each file consisting of a single document (=paper). 

Files are structured into folders by domain (e.g., ``ACL``) and by whether they originate in the full-text subset of MiST.
* In the ``fulltext`` parts, all modal instances in a given document have been annotated.
* In the ``non_fulltext`` parts, only selected sentences (i.e., not all sentences for all modals) have been annotated.

For each annotated modal instance, we additionally provide the preceding and the following sentence if applicable.

In addition, we provide complete documents for the Agr, CS, ES, and MS subcorpora in ``full_papers``.

For all papers, information on source, license,license/copyright notice, title, creators,journal and publisher can be found in ``metadata.csv``. The column ``full-text annotation`` indicates whether all modal instances in the given document have been annotated.

## File format
Sentences are separated by an empty line.
Each sentence starts with a meta-data line indicating its ID:

``#<sentence ID>``

Token lines are formatted as follows (with ``\t`` standing for a tab space):

``<token>\t<MiST label>\t<additional labels>``

Here, MiST label refers to the MiST labels as discussed in the paper.
We employ a multi-labeling scheme, in which multiple labels are separated by ``-``.
Additional labels are labels that we collected during the annotation, e.g., negation information, but did not use in the final annotation. We publish them alongside with the MiST labels to enable their re-use in future work.

## Corpus split
``split_info.csv`` details how the corpus was split for the experiments reported in the paper.
Instances from one document are always assigned to the same split.
* ``standard_train_test_split`` refers to the split we used unless otherwise mentioned. This split has a held-out test set of documents, indicated by ``standard_train_test_split:in_testset`` in the CSV file. For documents that are not part of the test set, the column ``standard_train_test_split:train_fold`` indicates to which fold a document belongs in the 5-fold CV on the training set (see paper).
* For the cross-domain experiments, we used a 6-fold CV as specified by the column ``cross_domain_experiments_split:fold`` in order to test on each document once.