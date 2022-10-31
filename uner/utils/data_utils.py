import os
from typing import Dict, List, Union

from torch.utils.data import Dataset

from uner.metainfo import Preprocessors
from .common_utils import import_dataset_builder_class


def get_labels(dataset: Union[Dataset, Dict[str, Dataset]]) -> List[str]:
    labels = []
    if isinstance(dataset, dict):
        for _dataset in dataset.values():
            labels.extend(get_labels(_dataset))
    else:
        for data in dataset:
            builder_name = dataset.info.builder_name
            builder_cls = import_dataset_builder_class(builder_name)
            labels.extend(builder_cls.parse_label(data))
    labels = sorted(set(labels))
    return labels


def gen_label2id(labels: List[str], mode) -> Dict[str, int]:
    if mode == Preprocessors.sequence_labeling_preprocessor:
        label2id = {'O': 0}
        for label in labels:
            label2id['B-' + label] = len(label2id)
            label2id['I-' + label] = len(label2id)
        return label2id
    elif mode == Preprocessors.global_pointer_preprocessor or mode == Preprocessors.multilabel_span_typing_preprocessor:
        label2id = {}
        for label in labels:
            label2id[label] = len(label2id)
        return label2id
    else:
        raise NotImplementedError('Preprocessor: ' + mode)
