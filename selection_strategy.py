# -*- coding: utf-8 -*-
import torch
import random
import numpy as np
import scipy
import copy as cp
from tqdm import tqdm
from collections import defaultdict
# from NCAT.NCAT import NCAT


def IRT(x, a, b):
    return 1/(1+np.exp(-a*(x-b)))


def Likelihood(x, y, a, b):
    return (a*(y-1/(1+np.exp(-a*(x-b))))).sum()


class MCMC_Selection(object):
    def __init__(self, train_data, test_data, concept_map, train_label, test_label, gamma, beta, params):
        self.device = torch.device("cuda" if (
            params.device == 'cuda' and torch.cuda.is_available()) else "cpu")
        self.train_data = train_data
        self.test_data = test_data
        self.concept_map = concept_map
        self.train_label = train_label
        self.test_label = test_label
        self.gamma = gamma
        self.beta = beta
        self.ga = torch.FloatTensor(gamma).to(self.device)
        self.be = torch.FloatTensor(beta).to(self.device)
        self.params = params
        # if self.params.method == 'ncat':
        #     self.model = NCAT(train_data.num_questions + 1, 256, 1, 0.1).to(self.device)
        #     self.model.load_state_dict(torch.load('model/' + self.params.data_name + "/mcmc/best_model.pth"))
        #     self.model.eval()

    # IRT prob
    def P(self, theta):
        return 1/(1+torch.exp(-self.ga*(theta-self.be)))

    # CCAT method
    def get_ccat(self, selected, theta, stu):
        with torch.no_grad():
            Pt = self.P(theta)
            Ptheta = torch.sigmoid(
                (self.ga*(self.train_label-self.test_label[stu])*selected).sum(-1)).reshape(-1, 1)
            F = self.ga*(Ptheta*(1-Pt)*self.train_label +
                         (1-Ptheta)*Pt*(1-self.train_label))
            return F.sum(0).data.cpu().numpy()

    # FSI
    def get_Fisher(self, theta):
        with torch.no_grad():
            Pt = self.P(theta)
            F = self.ga*self.ga*Pt*(1-Pt)
            return F.data.cpu().numpy()

    # KLI
    def get_kli(self, theta, untested, n):
        if n == 0:
            return np.random.choice(untested)
        max = 0
        max_index = -1
        for i in untested:
            a = self.gamma[i]
            b = self.beta[i]
            pred_estimate = IRT(theta, a, b)

            def kli(x):
                pred = a*(x-b)
                pred = 1 / (1 + np.exp(-pred))
                q_estimate = 1 - pred_estimate
                q = 1 - pred
                return pred_estimate * np.log(pred_estimate / pred) + \
                    q_estimate * np.log((q_estimate / q))
            c = 3
            boundaries = [[theta - c / np.sqrt(n), theta + c / np.sqrt(n)]]
            v, err = scipy.integrate.quad(
                kli, boundaries[0][0], boundaries[0][1])
            if v > max:
                max = v
                max_index = i
        return max_index

    # MAAT
    def get_maat(self, theta, untested, a, b, labels):
        with torch.no_grad():
            Pt = self.P(theta)
            emcs = np.zeros(self.test_data.num_questions)-1
            for q in untested:
                a_m = a+[self.gamma[q]]
                b_m = b+[self.beta[q]]
                emc = 0
                for l in range(2):
                    labels_m = labels+[l]
                    x = scipy.optimize.root(Likelihood, 0, args=(
                        np.array(labels_m), np.array(a_m), np.array(b_m))).x[0]
                    if x > 4:
                        x = 4
                    if x < -4:
                        x = -4
                    emc += (Pt[q]*l+(1-Pt[q])*(1-l))*np.abs(x-theta)
                emcs[q] = emc
            q_list = np.argsort(emcs)[::-1][:20]
        return q_list

    def IWKC(self, selected):
        WKC = defaultdict(int)
        for q in selected:
            if q in self.concept_map:
                kcs = self.concept_map[q]
                if isinstance(kcs, int):
                    WKC[kcs] += 1
                else:
                    for kc in kcs:
                        WKC[kc] += 1
        return sum([cnt/(cnt+1) for cnt in WKC.values()])

    # NCAT
    def get_ncat(self, batch_0_question, batch_1_question, p_0_t, p_1_t):
        data = {"p_0_rec": batch_0_question, "p_1_rec": batch_1_question,
                "p_0_t": p_0_t, "p_1_t": p_1_t}
        return self.model.predict(data)[:, 1:].cpu().detach().numpy()

    # BECAT
    def bce_loss_derivative(self, pred, target):
        """ get bce_loss_derivative
        Args:
            pred: float,
            target: int,
        Returns:
            the derivative of bce_loss
        """
        derivative = (pred - target) / (pred * (1 - pred))
        return derivative

    def get_BE_weights(self, pred_all, ga):
        """ get BE matrix
        Args:
            pred_all: dict, the questions you want to sample and their probability
        Returns:
            the BE matrix weights
        """
        d = 100
        Pre_true = pred_all
        Pre_false = 1 - pred_all
        Der = pred_all*(1-pred_all)*ga

        gradients_theta1 = self.bce_loss_derivative(Pre_true, 0.0)*Der
        gradients_theta2 = self.bce_loss_derivative(Pre_true, 1.0)*Der
        diff_norm_00 = torch.abs(
            (Pre_false*gradients_theta1).reshape(1, -1)-(Pre_false*gradients_theta1).reshape(-1, 1))
        diff_norm_01 = torch.abs(
            (Pre_false*gradients_theta1).reshape(1, -1)-(Pre_true*gradients_theta2).reshape(-1, 1))
        diff_norm_10 = torch.abs(
            (Pre_true*gradients_theta2).reshape(1, -1)-(Pre_false*gradients_theta1).reshape(-1, 1))
        diff_norm_11 = torch.abs(
            (Pre_true*gradients_theta2).reshape(1, -1)-(Pre_true*gradients_theta2).reshape(-1, 1))
        Expect = diff_norm_00 + diff_norm_01 + diff_norm_10 + diff_norm_11
        return d-Expect

    def F_s_func(self, Sp_set, w_ij_matrix, sampled_elements):
        """ get F_s of the questions have been chosen
        Args:
            S_set:list , the questions have been chosen
            w_ij_matrix: dict, the weight matrix
        Returns:
            the F_s of the chosen questions
        """
        res = 0.0
        for i in range(len(sampled_elements)):
            q = sampled_elements[i]
            if (q not in sampled_elements[Sp_set]):
                mx = 0
                for j in Sp_set:
                    if w_ij_matrix[i][j] > mx:
                        mx = w_ij_matrix[i][j]
                res += mx
        return res

    def delta_q_S_t(self, question_id, pred_all, S_set, sampled_elements):
        """ get BECAT Questions weights delta
        Args:
            question_id: int, question id
            pred_all:dict, the untest questions and their probability
            S_set:dict, chosen questions
            sampled_elements:nparray, sampled set from untest questions
        Returns:
            delta_q: float, delta_q of questions id
        """

        Sp_set = (-1-np.arange(len(S_set)))
        b_array = np.array(S_set)
        sampled_elements = np.concatenate((sampled_elements, b_array), axis=0)
        if question_id not in sampled_elements:
            sampled_elements = np.append(sampled_elements, question_id)
            Sp_set = Sp_set - 1
        Sp_set = list(Sp_set)[::-1]
        sampled_dict = pred_all[sampled_elements]
        w_ij_matrix = self.get_BE_weights(
            sampled_dict, self.ga[sampled_elements])
        F_s = self.F_s_func(Sp_set, w_ij_matrix, sampled_elements)
        Sp_set.append(np.argwhere(sampled_elements == question_id)[0][0])
        F_sp = self.F_s_func(Sp_set, w_ij_matrix, sampled_elements)
        return F_sp - F_s

    def get_becat(self, selected, untested, theta):
        Pt = self.P(theta)
        tmplen = len(selected)
        sampled_elements = np.random.choice(untested, tmplen+5)
        untested_deltaq = [self.delta_q_S_t(
            qid, Pt, selected, sampled_elements).item() for qid in untested]
        q = untested[np.argmax(untested_deltaq)]
        return q

    # select question for students
    def get_question(self):
        selected_questions = []
        stu_theta = []
        np.random.seed(self.params.seed)
        random.seed(self.params.seed)
        for stu in tqdm(range(self.test_data.num_students)):
            a = []
            b = []
            labels = []
            theta = []

            # initial student's ability
            x = np.random.randn(1)[0]
            selected_question = []
            unselected_set = set(torch.where(
                self.test_label[stu] >= 0)[0].cpu().numpy())
            selected = torch.zeros(
                self.test_data.num_questions).to(self.device)
            if self.params.method == 'ncat':
                batch_0_question = np.zeros([1, 21])
                batch_1_question = np.zeros([1, 21])
                p_0_t = np.ones(1).astype('int')
                p_1_t = np.ones(1).astype('int')
            # each step t,select one question for student to answer
            for i in range(20):
                if self.params.method == 'ccat':
                    F = self.get_ccat(selected, x, stu)
                    unselected_questions = list(unselected_set)
                    q = unselected_questions[np.argmax(
                        F[list(unselected_set)])]
                if self.params.method == 'fsi':
                    F = self.get_Fisher(x)
                    unselected_questions = list(unselected_set)
                    q = unselected_questions[np.argmax(
                        F[list(unselected_set)])]
                if self.params.method == 'kli':
                    q = self.get_kli(x, list(unselected_set), i)
                if self.params.method == 'random':
                    q = random.choice(list(unselected_set))
                if self.params.method == 'maat':
                    q_list = self.get_maat(
                        x, list(unselected_set), a, b, labels)
                    q = q_list[np.argmax(
                        [self.IWKC(selected_question+[q]) for q in q_list])]
                if self.params.method == 'ncat':
                    Q = self.get_ncat(batch_0_question,
                                      batch_1_question, p_0_t, p_1_t)
                    unselected_questions = list(unselected_set)
                    q = unselected_questions[np.argmax(
                        Q[0][unselected_questions])]
                    if self.test_data.data[stu][q] == 0:
                        batch_0_question[0][p_0_t[0]] = q+1
                        p_0_t[0] += 1
                    else:
                        batch_1_question[0][p_1_t[0]] = q+1
                        p_1_t[0] += 1
                if self.params.method == 'becat':
                    q = self.get_becat(selected_question,
                                       list(unselected_set), x)
                a.append(self.gamma[q])
                b.append(self.beta[q])
                # get the answer for question q
                labels.append(self.test_data.data[stu][q])
                selected[q] = 1
                # estimate student's ability
                x = scipy.optimize.root(Likelihood, x, args=(
                    np.array(labels), np.array(a), np.array(b))).x[0]
                if x > 4:
                    x = 4
                if x < -4:
                    x = -4
                selected_question.append(q)
                unselected_set.remove(q)
                theta.append(x)
            selected_questions.append(selected_question)
            stu_theta.append(theta)
        return selected_questions, stu_theta


class GD_Selection(object):
    def __init__(self, train_data, test_data, concept_map, train_label, test_label, irt_model, params):
        self.device = torch.device(
            "cuda") if params.device == 'cuda' else torch.device("cuda")
        self.train_data = train_data
        self.test_data = test_data
        self.concept_map = concept_map
        self.train_label = train_label
        self.test_label = test_label
        self.params = params
        self.irt = irt_model
        self.ga = irt_model.alpha.data.flatten()
        self.be = irt_model.beta.data.flatten()
        if self.params.method == 'ncat':
            self.model = NCAT(train_data.num_questions+1,
                              256, 1, 0.1).to(self.device)
            self.model.load_state_dict(torch.load(
                'model/'+self.params.data_name+"/gd/best_model.pth"))
            self.model.eval()

    # IRT method
    def P(self, theta):
        return 1/(1+torch.exp(-self.ga*(theta-self.be)))

    # CCAT
    def get_ccat(self, selected, theta, stu):
        with torch.no_grad():
            Pt = self.P(theta)
            Ptheta = torch.sigmoid(
                (self.ga*(self.train_label-self.test_label[stu])*selected).sum(-1)).reshape(-1, 1)
            F = self.ga*(Ptheta*(1-Pt)*self.train_label +
                         (1-Ptheta)*Pt*(1-self.train_label))
            return F.sum(0).data.cpu().numpy()

    # FSI
    def get_Fisher(self, theta):
        with torch.no_grad():
            Pt = self.P(theta)
            F = self.ga*self.ga*Pt*(1-Pt)
            return F.data.cpu().numpy()

    # KLI
    def get_kli(self, theta, untested, n):
        if n == 0:
            return np.random.choice(untested)
        max = -np.inf
        max_index = -1
        pred_estimates = self.P(theta)
        for i in untested:
            a = self.ga[i].item()
            b = self.be[i].item()
            pred_estimate = pred_estimates[i].item()

            def kli(x):
                pred = a*(x-b)
                pred = 1 / (1 + np.exp(-pred))
                q_estimate = 1 - pred_estimate
                q = 1 - pred
                return pred_estimate * np.log(pred_estimate / pred) + \
                    q_estimate * np.log((q_estimate / q))
            c = 3
            boundaries = [[theta - c / np.sqrt(n), theta + c / np.sqrt(n)]]
            v, err = scipy.integrate.quad(
                kli, boundaries[0][0], boundaries[0][1])
            if v > max:
                max = v
                max_index = i
        return max_index

    # MAAT
    def get_maat(self, theta, untested, selected, stu):
        irt_maat = cp.deepcopy(self.irt)
        # with torch.no_grad():
        Pt = self.P(theta)
        emcs = np.zeros(self.test_data.num_questions)-1
        label = cp.deepcopy(self.test_label[stu])
        theta_s = cp.deepcopy(irt_maat.n_students)
        for q in untested:
            select = cp.deepcopy(selected)
            select[q] = 1
            emc = 0
            for l in range(2):
                label[q] = l
                irt_maat.n_students = cp.deepcopy(theta_s)
                irt_maat.get_maat(torch.where(select == 1)[0], label)
                x = irt_maat.n_students[0][0].item()
                emc += (Pt[q]*l+(1-Pt[q])*(1-l))*np.abs(x-theta_s[0][0].item())
            emcs[q] = emc
        q_list = np.argsort(emcs)[::-1][:20]
        return q_list

    def IWKC(self, selected):
        WKC = defaultdict(int)
        for q in selected:
            if q in self.concept_map:
                kcs = self.concept_map[q]
                if isinstance(kcs, int):
                    WKC[kcs] += 1
                else:
                    for kc in kcs:
                        WKC[kc] += 1
        return sum([cnt/(cnt+1) for cnt in WKC.values()])

    # BECAT
    def get_ncat(self, batch_0_question, batch_1_question, p_0_t, p_1_t):
        data = {"p_0_rec": batch_0_question, "p_1_rec": batch_1_question,
                "p_0_t": p_0_t, "p_1_t": p_1_t}
        return self.model.predict(data)[:, 1:].cpu().detach().numpy()

    # BECAT
    def bce_loss_derivative(self, pred, target):
        """ get bce_loss_derivative
        Args:
            pred: float,
            target: int,
        Returns:
            the derivative of bce_loss
        """
        derivative = (pred - target) / (pred * (1 - pred))
        return derivative

    def get_BE_weights(self, pred_all, ga):
        """ get BE matrix
        Args:
            pred_all: dict, the questions you want to sample and their probability
        Returns:
            the BE matrix weights
        """
        d = 100
        Pre_true = pred_all
        Pre_false = 1 - pred_all
        Der = pred_all*(1-pred_all)*ga

        gradients_theta1 = self.bce_loss_derivative(Pre_true, 0.0)*Der
        gradients_theta2 = self.bce_loss_derivative(Pre_true, 1.0)*Der
        diff_norm_00 = torch.abs(
            (Pre_false*gradients_theta1).reshape(1, -1)-(Pre_false*gradients_theta1).reshape(-1, 1))
        diff_norm_01 = torch.abs(
            (Pre_false*gradients_theta1).reshape(1, -1)-(Pre_true*gradients_theta2).reshape(-1, 1))
        diff_norm_10 = torch.abs(
            (Pre_true*gradients_theta2).reshape(1, -1)-(Pre_false*gradients_theta1).reshape(-1, 1))
        diff_norm_11 = torch.abs(
            (Pre_true*gradients_theta2).reshape(1, -1)-(Pre_true*gradients_theta2).reshape(-1, 1))
        Expect = diff_norm_00 + diff_norm_01 + diff_norm_10 + diff_norm_11

        return d-Expect

    def F_s_func(self, Sp_set, w_ij_matrix, sampled_elements):
        """ get F_s of the questions have been chosen
        Args:
            S_set:list , the questions have been chosen
            w_ij_matrix: dict, the weight matrix
        Returns:
            the F_s of the chosen questions
        """
        res = torch.zeros(1).to(self.device)
        for i in range(len(sampled_elements)):
            q = sampled_elements[i]
            # print(sampled_elements[Sp_set])
            if (q not in sampled_elements[Sp_set]):
                mx = 0
                for j in Sp_set:
                    if w_ij_matrix[i][j] > mx:
                        mx = w_ij_matrix[i][j]
                res += mx
        return res

    def delta_q_S_t(self, question_id, pred_all, S_set, sampled_elements):
        """ get BECAT Questions weights delta
        Args:
            question_id: int, question id
            pred_all:dict, the untest questions and their probability
            S_set:dict, chosen questions
            sampled_elements:nparray, sampled set from untest questions
        Returns:
            delta_q: float, delta_q of questions id
        """

        Sp_set = (-1-np.arange(len(S_set)))
        b_array = np.array(S_set)
        sampled_elements = np.concatenate((sampled_elements, b_array), axis=0)
        if question_id not in sampled_elements:
            sampled_elements = np.append(sampled_elements, question_id)
            Sp_set = Sp_set - 1
        Sp_set = list(Sp_set)[::-1]
        sampled_dict = pred_all[sampled_elements]
        w_ij_matrix = self.get_BE_weights(
            sampled_dict, self.ga[sampled_elements])
        F_s = self.F_s_func(Sp_set, w_ij_matrix, sampled_elements)
        Sp_set.append(np.argwhere(sampled_elements == question_id)[0][0])
        F_sp = self.F_s_func(Sp_set, w_ij_matrix, sampled_elements)
        return F_sp - F_s

    def get_becat(self, selected, untested, theta):
        Pt = self.P(theta)
        tmplen = len(selected)
        sampled_elements = np.random.choice(untested, tmplen+5)
        untested_deltaq = [self.delta_q_S_t(
            qid, Pt, selected, sampled_elements).item() for qid in untested]
        q = untested[np.argmax(untested_deltaq)]
        return q

    # select question for students
    def get_question(self):
        selected_questions = []
        stu_theta = []
        np.random.seed(self.params.seed)
        random.seed(self.params.seed)
        self.irt.alpha.requires_grad = False
        self.irt.beta.requires_grad = False
        for stu in tqdm(range(self.test_data.num_students)):
            self.irt.n_students.data = torch.zeros(
                [self.train_data.num_students, 1]).to(self.device)
            theta = []
            # initial student's ability
            x = np.random.randn(1)[0]
            selected_question = []
            unselected_set = set(torch.where(
                self.test_label[stu] >= 0)[0].cpu().numpy())
            selected = torch.zeros(
                self.test_data.num_questions).to(self.device)
            if self.params.method == 'ncat':
                batch_0_question = np.zeros([1, 21])
                batch_1_question = np.zeros([1, 21])
                p_0_t = np.ones(1).astype('int')
                p_1_t = np.ones(1).astype('int')
            # for each step t, select one question for student to answer
            for i in range(20):
                if self.params.method == 'ccat':
                    # F=get_Rank(train_label,x)
                    F = self.get_ccat(selected, x, stu)
                    unselected_questions = list(unselected_set)
                    q = unselected_questions[np.argmax(
                        F[list(unselected_set)])]
                if self.params.method == 'fsi':
                    F = self.get_Fisher(x)
                    unselected_questions = list(unselected_set)
                    q = unselected_questions[np.argmax(
                        F[list(unselected_set)])]
                if self.params.method == 'kli':
                    q = self.get_kli(x, list(unselected_set), i)
                if self.params.method == 'random':
                    q = random.choice(list(unselected_set))
                if self.params.method == 'maat':
                    q_list = self.get_maat(
                        x, list(unselected_set), selected, stu)
                    q = q_list[np.argmax(
                        [self.IWKC(selected_question+[q]) for q in q_list])]
                if self.params.method == 'ncat':
                    Q = self.get_ncat(batch_0_question,
                                      batch_1_question, p_0_t, p_1_t)
                    unselected_questions = list(unselected_set)
                    q = unselected_questions[np.argmax(
                        Q[0][unselected_questions])]
                    if self.test_data.data[stu][q] == 0:
                        batch_0_question[0][p_0_t[0]] = q+1
                        p_0_t[0] += 1
                    else:
                        batch_1_question[0][p_1_t[0]] = q+1
                        p_1_t[0] += 1
                if self.params.method == 'becat':
                    q = self.get_becat(selected_question,
                                       list(unselected_set), x)
                selected[q] = 1
                selected_question.append(q)
                unselected_set.remove(q)
                # estimate student's ability
                self.irt.optim(torch.where(selected == 1)
                               [0], self.test_label[stu])
                x = self.irt.get_theta()
                # get student's current ability
                theta.append(x)
            selected_questions.append(selected_question)
            stu_theta.append(theta)
        return selected_questions, stu_theta
