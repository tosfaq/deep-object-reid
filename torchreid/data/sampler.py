from __future__ import division, absolute_import
import copy
import numpy as np
import random
from collections import defaultdict
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler

AVAI_SAMPLERS = ['RandomIdentitySampler', 'RandomIdentitySamplerV2', 'RandomIdentitySamplerV3',
                 'SequentialSampler', 'RandomSampler']


def build_train_sampler(data_source, train_sampler, batch_size=32, num_instances=4, **kwargs):
    """Builds a training sampler.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid).
        train_sampler (str): sampler name (default: ``RandomSampler``).
        batch_size (int, optional): batch size. Default is 32.
        num_instances (int, optional): number of instances per identity in a
            batch (when using ``RandomIdentitySampler``). Default is 4.
    """
    assert train_sampler in AVAI_SAMPLERS, \
        'train_sampler must be one of {}, but got {}'.format(AVAI_SAMPLERS, train_sampler)

    if train_sampler == 'RandomIdentitySampler':
        sampler = RandomIdentitySampler(data_source, batch_size, num_instances)
    elif train_sampler == 'RandomIdentitySamplerV2':
        sampler = RandomIdentitySamplerV2(data_source, batch_size, num_instances)
    elif train_sampler == 'RandomIdentitySamplerV3':
        sampler = RandomIdentitySamplerV3(data_source, batch_size, num_instances)
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
        self.index_dic = defaultdict(list)
        for index, record in enumerate(self.data_source):
            self.index_dic[record[1]].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        # TODO: improve precision
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(
                    idxs, size=self.num_instances, replace=True
                )
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomIdentitySamplerV2(RandomIdentitySampler):
    def __init__(self, data_source, batch_size, num_instances):
        super().__init__(data_source, batch_size, num_instances)

    def __iter__(self):
        random.shuffle(self.pids)
        output_ids = []
        for pid in self.pids:
            random.shuffle(self.index_dic[pid])
            output_ids += self.index_dic[pid]
        extra_samples = len(output_ids) % self.num_instances
        output_ids = output_ids[: len(output_ids) - extra_samples]
        ids = np.array(output_ids)
        ids = ids.reshape((-1, self.num_instances))
        np.random.shuffle(ids)
        ids = ids.reshape((-1))
        return iter(ids.tolist())

    def __len__(self):
        return len(self.data_source)


class RandomIdentitySamplerV3(Sampler):
    def __init__(self, data_source, batch_size, num_instances):
        super().__init__(data_source)

        if batch_size < num_instances:
            raise ValueError('batch_size={} must be no less than num_instances={}'
                             .format(batch_size, num_instances))

        self.num_instances = num_instances

        self.index_dict = dict()
        for index, record in enumerate(data_source):
            trg_name = record[3]
            if trg_name not in self.index_dict:
                self.index_dict[trg_name] = defaultdict(list)

            self.index_dict[trg_name][record[1]].append(index)

        self.pids = {trg_name: list(trg_dict.keys()) for trg_name, trg_dict in self.index_dict.items()}

        num_indices = [len(indices) for trg_dict in self.index_dict.values() for indices in trg_dict.values()]
        average_num_instances = np.median(num_indices)
        print('Median number of samples per ID: {}'.format(average_num_instances))

        self.num_packages = int(average_num_instances) // self.num_instances
        self.instances_per_pid = self.num_packages * self.num_instances
        self.length = sum([len(ids) for ids in self.pids.values()]) * self.instances_per_pid

    def __iter__(self):
        final_pids = [(trg_name, trg_pid) for trg_name, trg_pids in self.pids.items() for trg_pid in trg_pids]
        random.shuffle(final_pids)

        final_data_indices = []
        for trg_name, trg_pid in final_pids:
            sampled_indices = copy.deepcopy(self.index_dict[trg_name][trg_pid])
            random.shuffle(sampled_indices)

            if len(sampled_indices) < self.instances_per_pid:
                num_extra = self.instances_per_pid - len(sampled_indices)
                extra_indices = np.random.choice(sampled_indices, size=num_extra, replace=True)

                sampled_indices.extend(extra_indices)
                random.shuffle(sampled_indices)
            elif len(sampled_indices) > self.instances_per_pid:
                sampled_indices = sampled_indices[:self.instances_per_pid]

            final_data_indices.append(sampled_indices)

        final_data_indices = np.array(final_data_indices).reshape((len(final_pids), -1, self.num_instances))
        final_data_indices = np.transpose(final_data_indices, (1, 0, 2)).reshape(-1)

        return iter(final_data_indices.tolist())

    def __len__(self):
        return self.length
