from __future__ import division, absolute_import
import copy
import numpy as np
import random
from collections import defaultdict
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler

AVAI_SAMPLERS = ['RandomIdentitySampler', 'RandomIdentitySamplerV2', 'RandomIdentitySamplerV3',
                 'SequentialSampler', 'RandomSampler']


def build_train_sampler(data_source, train_sampler, batch_size=32, batch_num_instances=4,
                        epoch_num_instances=-1, fill_instances=False, **kwargs):
    """Builds a training sampler.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid).
        train_sampler (str): sampler name (default: ``RandomSampler``).
        batch_size (int, optional): batch size. Default is 32.
        batch_num_instances (int, optional): number of instances per identity in a
            batch (when using ``RandomIdentitySampler``). Default is 4.
        epoch_num_instances (int, optional): number of instances per epoch
            (when using ``RandomIdentitySamplerV3``). Default is -1 (auto configuration).
    """
    assert train_sampler in AVAI_SAMPLERS, \
        'train_sampler must be one of {}, but got {}'.format(AVAI_SAMPLERS, train_sampler)

    if train_sampler == 'RandomIdentitySampler':
        sampler = RandomIdentitySampler(data_source, batch_size, batch_num_instances)
    elif train_sampler == 'RandomIdentitySamplerV2':
        sampler = RandomIdentitySamplerV2(data_source, batch_size, batch_num_instances, fill_instances)
    elif train_sampler == 'RandomIdentitySamplerV3':
        sampler = RandomIdentitySamplerV3(data_source, batch_size, batch_num_instances, epoch_num_instances)
    elif train_sampler == 'SequentialSampler':
        sampler = SequentialSampler(data_source)
    elif train_sampler == 'RandomSampler':
        sampler = RandomSampler(data_source)
    else:
        raise ValueError('Unknown sampler: {}'.format(train_sampler))

    return sampler


class RandomIdentitySampler(Sampler):
    """Randomly samples N identities each with K instances.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid).
        batch_size (int): batch size.
        num_instances (int): number of instances per identity in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        if batch_size < num_instances:
            raise ValueError(
                'batch_size={} must be no less '
                'than num_instances={}'.format(batch_size, num_instances)
            )

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances

        self.index_dict = dict()
        for index, record in enumerate(data_source):
            trg_name = record[3] if len(record) > 3 else 0
            if trg_name not in self.index_dict:
                self.index_dict[trg_name] = defaultdict(list)

            obj_id = record[1]
            self.index_dict[trg_name][obj_id].append(index)
        self.pids = {trg_name: list(trg_dict.keys()) for trg_name, trg_dict in self.index_dict.items()}

        # estimate number of examples in an epoch
        # TODO: improve precision
        self.length = 0
        for pids in self.index_dict.values():
            for idxs in pids.values():
                num = len(idxs)
                if num < self.num_instances:
                    num = self.num_instances
                self.length += num - num % self.num_instances

    def __iter__(self):
        final_pids = [(trg_name, trg_pid) for trg_name, trg_pids in self.pids.items() for trg_pid in trg_pids]
        random.shuffle(final_pids)

        batch_idxs_dict = {trg_name: defaultdict(list) for trg_name in self.pids.keys()}
        for trg_name, trg_pid in final_pids:
            idxs = copy.deepcopy(self.index_dict[trg_name][trg_pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)

            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[trg_name][trg_pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = [(trg_name, trg_pid) for trg_name, trg_pids in self.pids.items() for trg_pid in trg_pids]

        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for trg_name, trg_pid in selected_pids:
                batch_idxs = batch_idxs_dict[trg_name][trg_pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[trg_name][trg_pid]) == 0:
                    avai_pids.remove((trg_name, trg_pid))

        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomIdentitySamplerV2(RandomIdentitySampler):
    def __init__(self, data_source, batch_size, num_instances, fill_instances=False):
        super().__init__(data_source, batch_size, num_instances)

        self.fill_instances = fill_instances
        if self.fill_instances:
            self.length = sum([len(trg_idxs) + len(trg_idxs) % self.num_instances
                               for trg_pids in self.index_dict.values() for trg_idxs in trg_pids.values()])
        else:
            self.length = len(self.data_source) - len(self.data_source) % self.num_instances

    def __iter__(self):
        final_pids = [(trg_name, trg_pid) for trg_name, trg_pids in self.pids.items() for trg_pid in trg_pids]
        random.shuffle(final_pids)

        output_ids = []
        for trg_name, trg_pid in final_pids:
            candidates = self.index_dict[trg_name][trg_pid]
            random.shuffle(candidates)

            output_ids += candidates
            if self.fill_instances and len(candidates) % self.num_instances > 0:
                extra_size = len(candidates) % self.num_instances
                output_ids += np.random.choice(candidates, size=extra_size, replace=True).tolist()

        num_rest_samples = len(output_ids) % self.num_instances
        output_ids = output_ids[: len(output_ids) - num_rest_samples]

        ids = np.array(output_ids)
        ids = ids.reshape((-1, self.num_instances))
        np.random.shuffle(ids)
        ids = ids.reshape((-1))

        return iter(ids.tolist())


class RandomIdentitySamplerV3(Sampler):
    def __init__(self, data_source, batch_size, num_instances, epoch_num_instances=-1):
        if batch_size < num_instances:
            raise ValueError('batch_size={} must be no less than num_instances={}'
                             .format(batch_size, num_instances))

        self.num_instances = num_instances

        self.index_dict = dict()
        for index, record in enumerate(data_source):
            trg_name = record[3] if len(record) > 3 else 0
            if trg_name not in self.index_dict:
                self.index_dict[trg_name] = defaultdict(list)

            obj_id = record[1]
            self.index_dict[trg_name][obj_id].append(index)
        self.orig_index_dict = copy.deepcopy(self.index_dict)

        self.pids = {trg_name: list(trg_dict.keys()) for trg_name, trg_dict in self.index_dict.items()}

        if epoch_num_instances is not None and epoch_num_instances > 0:
            average_num_instances = epoch_num_instances
            print('[INFO] Manually set the number of samples per ID for epoch: {}'.format(average_num_instances))
        else:
            num_indices = [len(indices) for trg_dict in self.index_dict.values() for indices in trg_dict.values()]
            average_num_instances = np.median(num_indices)
            print('[INFO] Estimated the number of samples per ID for epoch: {}'.format(average_num_instances))

        self.num_packages = int(average_num_instances) // self.num_instances
        self.instances_per_pid = self.num_packages * self.num_instances
        if self.instances_per_pid != int(average_num_instances):
            print('[INFO] Forced the number of samples per ID for epoch: {}'.format(self.instances_per_pid))

        self.length = sum([len(ids) for ids in self.pids.values()]) * self.instances_per_pid

    def __iter__(self):
        final_pids = [(trg_name, trg_pid) for trg_name, trg_pids in self.pids.items() for trg_pid in trg_pids]
        random.shuffle(final_pids)

        final_data_indices = []
        for trg_name, trg_pid in final_pids:
            rest_indices = self.index_dict[trg_name][trg_pid]
            random.shuffle(rest_indices)

            if len(rest_indices) < self.instances_per_pid:
                sampled_indices = copy.deepcopy(rest_indices)
                total_rest_num = self.instances_per_pid - len(rest_indices)
                while True:
                    rest_indices = copy.deepcopy(self.orig_index_dict[trg_name][trg_pid])
                    if len(rest_indices) <= total_rest_num:
                        sampled_indices.extend(rest_indices)
                        total_rest_num -= len(rest_indices)
                    else:
                        random.shuffle(rest_indices)
                        sampled_indices.extend(copy.deepcopy(rest_indices[:total_rest_num]))

                        rest_indices = rest_indices[total_rest_num:]
                        self.index_dict[trg_name][trg_pid] = rest_indices

                        break

                random.shuffle(sampled_indices)
            else:
                sampled_indices = copy.deepcopy(rest_indices[:self.instances_per_pid])

                rest_indices = rest_indices[self.instances_per_pid:]
                self.index_dict[trg_name][trg_pid] = rest_indices

            final_data_indices.append(sampled_indices)

        final_data_indices = np.array(final_data_indices).reshape((len(final_pids), -1, self.num_instances))
        final_data_indices = np.transpose(final_data_indices, (1, 0, 2)).reshape(-1)

        return iter(final_data_indices.tolist())

    def __len__(self):
        return self.length
