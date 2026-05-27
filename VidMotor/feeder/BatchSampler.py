from torch.utils.data import Sampler
import random
import numpy as np


class SubjectBatchSampler(Sampler):
    """
    A PyTorch Sampler that generates batches based on subjects instead of individual samples.
    Each batch contains all samples from a fixed number of subjects.
    """
    def __init__(self, dataset, subjects_per_batch, shuffle=True, drop_last=True):
        self.dataset = dataset
        self.subjects_per_batch = subjects_per_batch
        self.shuffle = shuffle
        self.drop_last = drop_last
        # Create a mapping from subject to sample indices
        self.subject_to_indices = {}
        for idx, subject in enumerate(self.dataset.subject_name):
            if subject not in self.subject_to_indices:
                self.subject_to_indices[subject] = []
            # Keys are subjects, values are lists of all sample indices belonging to each subject
            self.subject_to_indices[subject].append(idx)
        self.subjects = list(self.subject_to_indices.keys())
        for sub in self.subjects:
            assert len(self.subject_to_indices[sub]) == self.dataset.subject_name.count(sub)

    def __iter__(self):
        subjects = self.subjects.copy()
        if self.shuffle:
            random.shuffle(subjects)
        for i in range(0, len(self.subjects), self.subjects_per_batch):
            selected_subjects = subjects[i:i + self.subjects_per_batch]
            if self.drop_last and len(selected_subjects) < self.subjects_per_batch:
                break
            batch_indices = np.concatenate([self.subject_to_indices[subject] for subject in selected_subjects]).tolist()
            yield batch_indices

    def __len__(self):
        if self.drop_last:
            return len(self.subjects) // self.subjects_per_batch
        else:
            return (len(self.subjects) + self.subjects_per_batch - 1) // self.subjects_per_batch
