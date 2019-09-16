import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np


class GE2ELoss(nn.Module):

    def __init__(self, init_w=10.0, init_b=-5.0, loss_method='softmax'):
        '''
        Implementation of the Generalized End-to-End loss defined in https://arxiv.org/abs/1710.10467 [1]
        Accepts an input of size (N, M, D)
            where N is the number of speakers in the batch,
            M is the number of utterances per speaker,
            and D is the dimensionality of the embedding vector (e.g. d-vector)
        Args:
            - init_w (float): defines the initial value of w in Equation (5) of [1]
            - init_b (float): definies the initial value of b in Equation (5) of [1]
        '''
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.loss_method = loss_method

        assert self.loss_method in ['softmax', 'contrast']

        if self.loss_method == 'softmax':
            self.embed_loss = self.embed_loss_softmax
        if self.loss_method == 'contrast':
            self.embed_loss = self.embed_loss_contrast

        self.cos_sim_matrix = None

    def calc_new_centroids(self, dvecs, centroids, spkr, utt):
        '''
        Calculates the new centroids excluding the reference utterance
        '''
        excl = torch.cat((dvecs[spkr,:utt], dvecs[spkr,utt+1:]))
        excl = torch.mean(excl, 0)
        new_centroids = []
        for i, centroid in enumerate(centroids):
            if i == spkr:
                new_centroids.append(excl)
            else:
                new_centroids.append(centroid)
        return torch.stack(new_centroids)

    def calc_cosine_sim(self, dvecs, centroids):
        '''
        Make the cosine similarity matrix with dims (N,M,N)
        '''
        cos_sim_matrix = []
        for spkr_idx, speaker in enumerate(dvecs):
            cs_row = []
            for utt_idx, utterance in enumerate(speaker):
                new_centroids = self.calc_new_centroids(dvecs, centroids, spkr_idx, utt_idx)
                # vector based cosine similarity for speed
                cs_row.append(torch.clamp(torch.mm(utterance.unsqueeze(1).transpose(0,1), new_centroids.transpose(0,1)) / (torch.norm(utterance) * torch.norm(new_centroids, dim=1)), 1e-6))
            cs_row = torch.cat(cs_row, dim=0)
            cos_sim_matrix.append(cs_row)
        return torch.stack(cos_sim_matrix)

    def embed_loss_softmax(self, dvecs, cos_sim_matrix):
        '''
        Calculates the loss on each embedding $L(e_{ji})$ by taking softmax
        '''
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                L_row.append(-F.log_softmax(cos_sim_matrix[j,i], 0)[j])
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    def embed_loss_contrast(self, dvecs, cos_sim_matrix):
        '''
        Calculates the loss on each embedding $L(e_{ji})$ by contrast loss with closest centroid
        '''
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                centroids_sigmoids = torch.sigmoid(cos_sim_matrix[j,i])
                excl_centroids_sigmoids = torch.cat((centroids_sigmoids[:j], centroids_sigmoids[j+1:]))
                L_row.append(1. - torch.sigmoid(cos_sim_matrix[j,i,j]) + torch.max(excl_centroids_sigmoids))
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    def forward(self, dvecs):
        '''
        Calculates the GE2E loss for an input of dimensions (num_speakers, num_utts_per_speaker, dvec_feats)
        '''
        #Calculate centroids
        centroids = torch.mean(dvecs, 1)

        #Calculate the cosine similarity matrix
        cos_sim_matrix = self.calc_cosine_sim(dvecs, centroids)
        self.cos_sim_matrix = cos_sim_matrix
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        L = self.embed_loss(dvecs, cos_sim_matrix)
        return L.sum()

    def compute_eer(self, dvecs):
        with torch.no_grad():
            speakers_per_batch, utterances_per_speaker = dvecs.shape[:2]
            ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])

            centroids = torch.mean(dvecs, 1)
            cos_sim_matrix = self.calc_cosine_sim(dvecs, centroids)
            cos_sim_matrix = cos_sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, speakers_per_batch))
            preds = cos_sim_matrix.detach().cpu().numpy()
            # Snippet from https://yangcha.github.io/EER-ROC/
            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return eer


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        return self.loss_fn(outputs, targets)

    def calc_new_centroids(self, dvecs, centroids, spkr, utt):
        '''
        Calculates the new centroids excluding the reference utterance
        '''
        excl = torch.cat((dvecs[spkr,:utt], dvecs[spkr,utt+1:]))
        excl = torch.mean(excl, 0)
        new_centroids = []
        for i, centroid in enumerate(centroids):
            if i == spkr:
                new_centroids.append(excl)
            else:
                new_centroids.append(centroid)
        return torch.stack(new_centroids)

    def calc_cosine_sim(self, dvecs, centroids):
        '''
        Make the cosine similarity matrix with dims (N,M,N)
        '''
        cos_sim_matrix = []
        for spkr_idx, speaker in enumerate(dvecs):
            cs_row = []
            for utt_idx, utterance in enumerate(speaker):
                new_centroids = self.calc_new_centroids(dvecs, centroids, spkr_idx, utt_idx)
                # vector based cosine similarity for speed
                cs_row.append(torch.clamp(torch.mm(utterance.unsqueeze(1).transpose(0,1), new_centroids.transpose(0,1)) / (torch.norm(utterance) * torch.norm(new_centroids, dim=1)), 1e-6))
            cs_row = torch.cat(cs_row, dim=0)
            cos_sim_matrix.append(cs_row)
        return torch.stack(cos_sim_matrix)

    def compute_eer(self, dvecs):
        with torch.no_grad():
            speakers_per_batch, utterances_per_speaker = dvecs.shape[:2]
            ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])

            centroids = torch.mean(dvecs, 1)
            cos_sim_matrix = self.calc_cosine_sim(dvecs, centroids)
            cos_sim_matrix = cos_sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, speakers_per_batch))
            preds = cos_sim_matrix.detach().cpu().numpy()
            # Snippet from https://yangcha.github.io/EER-ROC/
            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return eer
