from typing import Optional, Union, Dict, Tuple, List, Any
import os
import json

import numpy as np
import tensorflow as tf

from quickstats import AbstractObject, timer, VerbosePrint
from quickstats.utils.common_utils import execute_multi_tasks
from quickstats.utils.string_utils import get_field_names
from .dataset import write_tfrecord

class TFRecordMaker(AbstractObject):

    def __init__(self, filename:str,
                 data: Dict[str, np.ndarray],
                 num_shards:int=1,
                 metadata_filename:Optional[str]=None,
                 save_metadata:bool=True,
                 cache:bool=True,
                 parallel:int=-1,
                 verbosity:Optional[Union[int, str]]="INFO"):
        super().__init__(verbosity=verbosity)
        self.data = data
        self.filename = filename
        self.num_shards = num_shards
        self.save_metadata = save_metadata
        self.metadata_filename = metadata_filename
        self.cache = cache
        self.parallel = parallel    

    def prepare_task_inputs(self) -> List[Dict]:
        
        shard_indices = np.arange(self.num_shards)

        def get_sharded_filename(filename:str):
            field_names = get_field_names(filename)
            if 'shard_index' not in field_names:
                root, ext = os.path.splitext(filename)
                return f"{root}_{{shard_index}}{ext}"
            return filename

        filename = self.filename
        if self.num_shards > 1:
            filename = get_sharded_filename(filename)

        metadata_filename = self.metadata_filename
        if self.save_metadata:
            if metadata_filename is None:
                root, ext = os.path.splitext(filename)
                metadata_filename = f"{root}_metadata.json"
            if self.num_shards > 1:
                metadata_filename = get_sharded_filename(metadata_filename)
        else:
            metadata_filename = None

        data = self.data
        data_splits = {}
        for key, array in data.items():
            data_splits[key] = np.array_split(array, self.num_shards)

        kwargs_list = []
        cached_filenames = []
        for i in shard_indices:
            filename_i = filename.format(shard_index=i)
            if metadata_filename:
                metadata_filename_i = metadata_filename.format(shard_index=i)
            else:
                metadata_filename_i = None
            if self.cache and os.path.exists(filename_i) and \
            ((not self.save_metadata) or os.path.exists(metadata_filename_i)):
                cached_filenames.append(filename_i)
                continue
            kwargs = {
                'data': {key: arrays[i] for key, arrays in data_splits.items()},
                'filename': filename_i,
                'metadata_filename': metadata_filename_i,
                'verbosity': self.stdout.verbosity,
            }
            kwargs_list.append(kwargs)

        return kwargs_list, cached_filenames

    @staticmethod
    def run_instance(kwargs:Dict[str, Any]):
        filename = kwargs['filename']
        metadata_filename = kwargs['metadata_filename']
        stdout = VerbosePrint(kwargs['verbosity'])

        data = kwargs['data']
        with tf.io.TFRecordWriter(filename) as writer:
            metadata = write_tfrecord(writer, **data)
        stdout.info(f"Saved output as {filename}")
        
        if metadata_filename is not None:
            with open(metadata_filename, 'w') as file:
                json.dump(metadata, file)

    def run(self):
        kwargs_list, cached_filenames = self.prepare_task_inputs()
        for cached_filename in cached_filenames:
            self.stdout.info(f'Cached output from {cached_filename}')
        with timer() as t:
            execute_multi_tasks(self.run_instance, kwargs_list, parallel=self.parallel)
        self.stdout.info(f'Task finished. Total time taken: {t.interval:.3f} s.')