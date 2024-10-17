# -*- coding: utf-8 -*-
from collections import defaultdict, deque

class Dataset(object):

    def __init__(self, data, num_students, num_questions):
        """
        Args:
            data: list, [(sid, qid, score)]
            concept_map: dict, concept map {qid: cid}
            num_students: int, total student number
            num_questions: int, total question number
            num_concepts: int, total concept number
        """
        self._raw_data = data
        self.n_students = num_students
        self.n_questions = num_questions
        
        # reorganize datasets
        self._data = {}
        for sid, qid, correct in data:
            self._data.setdefault(sid, {})
            self._data[sid].setdefault(qid, {})
            self._data[sid][qid] = correct

        student_ids = set(x[0] for x in data)
        question_ids = set(x[1] for x in data)
        assert max(student_ids) < num_students, \
            'Require student ids renumbered'
        assert max(question_ids) < num_questions, \
            'Require student ids renumbered'

    @property
    def num_students(self):
        return self.n_students

    @property
    def num_questions(self):
        return self.n_questions

    @property
    def raw_data(self):
        return self._raw_data

    @property
    def data(self):
        return self._data
    
class TrainDataset(Dataset):

    def __init__(self, data, num_students, num_questions):
        """
        Args:
            data: list, [(sid, qid, score)]
            concept_map: dict, concept map {qid: cid}
            num_students: int, total student number
            num_questions: int, total question number
            num_concepts: int, total concept number
        """
        super().__init__(data, num_students, num_questions)

    def __getitem__(self, item):
        sid, qid, score = self.raw_data[item]
        return sid, qid, score

    def __len__(self):
        return len(self.raw_data)
    
class AdapTestDataset(Dataset):

    def __init__(self, data, num_students, num_questions):
        """
        Args:
            data: list, [(sid, qid, score)]
            concept_map: dict, concept map {qid: cid}
            num_students: int, total student number
            num_questions: int, total question number
            num_concepts: int, total concept number
        """
        super().__init__(data,num_students, num_questions)

        # initialize tested and untested set
        self._tested = None
        self._untested = None
        self.reset()

    def apply_selection(self, student_idx, question_idx):
        """ 
        Add one untested question to the tested set
        Args:
            student_idx: int
            question_idx: int
        """
        assert question_idx in self._untested[student_idx], \
            'Selected question not allowed'
        self._untested[student_idx].remove(question_idx)
        self._tested[student_idx].append(question_idx)

    def reset(self):
        """ 
        Set tested set empty
        """
        self._tested = defaultdict(deque)
        self._untested = defaultdict(set)
        for sid in self.data:
            self._untested[sid] = set(self.data[sid].keys())

    @property
    def tested(self):
        return self._tested

    @property
    def untested(self):
        return self._untested

    def get_tested_dataset(self, last=False):
        """
        Get tested data for training
        Args: 
            last: bool, True - the last question, False - all the tested questions
        Returns:
            TrainDataset
        """
        triplets = []
        for sid, qids in self._tested.items():
            if last:
                qid = qids[-1]
                triplets.append((sid, qid, self.data[sid][qid]))
            else:
                for qid in qids:
                    triplets.append((sid, qid, self.data[sid][qid]))
        return TrainDataset(triplets, self.num_students, self.num_questions)