import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import operator
import torch.nn.functional as F
import torch.utils.data as data_utils
from transformers import AutoTokenizer, AutoModel
import os
from sklearn.metrics import classification_report

class PrototypeLayer(nn.Module):
    def __init__(self, k_protos, vect_size):
        super(PrototypeLayer, self).__init__()
        self.n_protos = k_protos
        self.vect_size = vect_size
        self.prototypes = nn.Parameter(torch.Tensor(k_protos, vect_size))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, inputs):
        tmp1 = inputs.unsqueeze(2)
        tmp1 = tmp1.expand(tmp1.shape[0], tmp1.shape[1], self.n_protos, self.vect_size)
        tmp2 = self.prototypes.expand(tmp1.shape[0], tmp1.shape[1], self.n_protos, self.vect_size)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp3 * tmp3
        distances = tmp4.sum(dim=3)
        return distances, self.prototypes

class DistanceLayer(nn.Module):
    def __init__(self):
        super(DistanceLayer, self).__init__()
        self.a = 0.1
        self.beta = 1e6

    def e_func(self, x):
        return torch.exp(-(self.a * x))

    def forward(self, full_distances):
        min_dist_ind = F.softmax(-full_distances * self.beta, dim=2)
        e_dist = self.e_func(full_distances) + 1e-8
        dist_hot_vect = min_dist_ind * e_dist
        return dist_hot_vect

class CustomModel(nn.Module):
    def __init__(self, k_protos, vect_size):
        super(CustomModel, self).__init__()
        self.proto_layer = PrototypeLayer(k_protos, vect_size)
        self.distance_layer = DistanceLayer()

    def forward(self, inputs):
        full_distances, _ = self.proto_layer(inputs)
        dist_hot_vect = self.distance_layer(full_distances)
        return dist_hot_vect

class ProtoryNet:
    def __init__(self):
        self.mappedPrototypes = {}

    def createModel(self, k_cents, k_protos=10, vect_size=512, alpha=0.0001, beta=0.01):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased")
        
        class CustomDataset(data_utils.Dataset):
            def __init__(self, data, labels):
                self.data = data
                self.labels = labels

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]

        class CustomCollator:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer

            def __call__(self, batch):
                texts, labels = zip(*batch)
                encodings = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
                input_ids = encodings['input_ids']
                attention_mask = encodings['attention_mask']
                return input_ids, attention_mask, torch.tensor(labels)

        loss_tracker = nn.MSELoss(reduction='mean')

        self.proto_layer = PrototypeLayer(k_protos, vect_size)
        self.distance_layer = DistanceLayer()
        self.custom_model = CustomModel(k_protos, vect_size)

        optimizer = optim.Adam(self.custom_model.parameters(), lr=0.0001)

        train_dataset = CustomDataset(x_train, y_train)
        train_loader = data_utils.DataLoader(train_dataset, batch_size=32, collate_fn=CustomCollator(tokenizer))

        maxEvalRes = 0

        for epoch in range(100):
            print("Epoch ", epoch)
            for i, (input_ids, attention_mask, labels) in enumerate(train_loader):
                optimizer.zero_grad()

                input_ids = input_ids.squeeze(1)
                attention_mask = attention_mask.squeeze(1)

                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs = self.custom_model(input_ids)
                dist, _ = self.proto_layer(outputs)
                cost2 = torch.sum(torch.min(dist, dim=1).values)

                d = self.pw_distance(self.proto_layer.prototypes)
                diag_ones = torch.eye(k_protos, dtype=torch.float32).to(device)
                d1 = d + diag_ones * torch.max(d)
                d2 = torch.min(d1, dim=1).values
                min_d2_dist = torch.min(d2)
                cost3 = self.tight_pos_sigmoid_offset(min_d2_dist, 1) + 1e-8

                y_val = labels.unsqueeze(0)
                loss_object = nn.BCELoss()
                loss = loss_object(outputs, y_val) + alpha * cost2 + beta * cost3

                loss.backward()
                optimizer.step()
                loss_tracker.update(loss.item())

                if i % 200 == 0:
                    y_preds, score = self.evaluate(x_test, y_test)
                    print("Evaluate on valid set: ", score)
                    if score > maxEvalRes:
                        maxEvalRes = score
                        print("This is the best eval res, saving the model...")
                        now = datetime.now()
                        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
                        if saveModel:
                            torch.save(self.custom_model.state_dict(), 'my_model.pth')
                        print("Just saved")

    def pw_distance(self, A):
        r = torch.sum(A * A, 1)
        r = r.view(-1, 1)
        D = r - 2 * torch.matmul(A, A.t()) + r.t()
        return D

    def tight_pos_sigmoid_offset(self, x, offset):
        return 1 / (1 + torch.exp(1 * (offset * x - 0.5)))

    def evaluate(self, x_valid, y):
        right, wrong = 0, 0
        count = 0
        y_preds = []
        with torch.no_grad():
            for x, y in zip(x_valid, y):
                x = x.unsqueeze(0)
                outputs = self.custom_model(x)
                y_pred = outputs[0][0]
                y_preds.append(y_pred)
                if count % 500 == 0:
                    print('Evaluating y_pred, y ', y_pred, round(y_pred.item()), y)
                if round(y_pred.item()) == y:
                    right += 1
                else:
                    wrong += 1
                count += 1
        return y_preds, right / (right + wrong)
