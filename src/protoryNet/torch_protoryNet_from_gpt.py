import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import operator

class ProtoryNet(nn.Module):
    def __init__(self):
        super(ProtoryNet, self).__init__()
        self.mappedPrototype = {}

    # Create the ProtoryNet:
    # inputs:
    # +k_cents: the initialized values of prototypes. In the paper, we used KMedoids clustering
    #            to have these values
    # vect_size: the dimension of the embedded sentence space, if using Google Universal Encoder,
    #            this value is 512
    # alpha and beta: the parameters used in the paper, default values are .0001 and .01
    def createModel(self, k_cents, k_protos=10, vect_size=512, alpha=0.0001, beta=0.01):
        loss_tracker = nn.MSELoss()

        # Prototype layer
        class PrototypeLayer(nn.Module):
            def __init__(self, k_protos, vect_size):
                super(PrototypeLayer, self).__init__()
                self.n_protos = k_protos
                self.vect_size = vect_size
                self.prototypes = nn.Parameter(torch.Tensor(k_protos, vect_size))
                nn.init.constant_(self.prototypes, k_cents)

            def forward(self, inputs):
                tmp1 = inputs.unsqueeze(2)
                tmp1 = tmp1.expand(-1, -1, self.n_protos, self.vect_size)
                tmp2 = self.prototypes.expand(tmp1.shape)
                tmp3 = tmp1 - tmp2
                tmp4 = tmp3 * tmp3
                distances = tmp4.sum(dim=3)
                return distances, self.prototypes

        # Distance layer: to convert the full distance matrix to sparse similarity matrix
        class DistanceLayer(nn.Module):
            def __init__(self):
                super(DistanceLayer, self).__init__()
                self.a = 0.1
                self.beta = 1e6

            def e_func(self, x):
                return torch.exp(-(self.a * x))

            def forward(self, full_distances):
                min_dist_ind = F.softmax(-full_distances * self.beta, dim=-1)
                e_dist = self.e_func(full_distances) + 1e-8
                dist_hot_vect = min_dist_ind * e_dist
                return dist_hot_vect

        # Customized model
        class CustomModel(nn.Module):
            def __init__(self, k_protos):
                super(CustomModel, self).__init__()
                self.k_protos = k_protos
                self.proto_layer = PrototypeLayer(k_protos, vect_size)
                self.distance_layer = DistanceLayer()

            def forward(self, x):
                dist, prototypes = self.proto_layer(x)
                return dist, prototypes

        inputLayer = nn.Sequential(
            nn.Embedding(1, vect_size),  # Assuming a single input sentence
            nn.Flatten(),
        )

        self.proto_layer = PrototypeLayer(k_protos, vect_size)
        self.distance_layer = DistanceLayer()

        RNN_CELL_SIZE = 128
        lstmop, (forward_h, forward_c) = nn.LSTM(vect_size, RNN_CELL_SIZE, batch_first=True, return_sequences=True, return_state=True)(self.distance_layer.full_distances)

        z1 = nn.Sequential(
            nn.Linear(RNN_CELL_SIZE, 1),
            nn.Sigmoid()
        )

        model = CustomModel(k_protos)

        for l in model.children():
            if isinstance(l, PrototypeLayer):
                protoLayerName = l
            if isinstance(l, DistanceLayer):
                distanceLayerName = l

        protoLayer = protoLayerName
        distLayer = distanceLayerName

        print("[db] model.input = ", inputLayer)
        print("[db] protoLayerName = ", protoLayerName)
        print("[db] protoLayer = ", protoLayer)
        print("[db] protoLayer.output = ", protoLayer.prototypes)
        print("[db] distanceLayer.output = ", distLayer)

        auxModel = nn.Sequential(
            inputLayer,
            protoLayer,
        )

        auxModel1 = nn.Sequential(
            inputLayer,
            distLayer,
        )

        auxModel2 = nn.Sequential(
            inputLayer,
            model,
        )

        model.auxModel = auxModel
        model.auxModel1 = auxModel1
        self.auxModel2 = auxModel2
        self.embModel = nn.Sequential(
            inputLayer,
            nn.Identity()
        )

        self.model = model
        return model

    def embed(self, input):
        return self.embModel(input)

    # Evaluate the model performance on the validation set
    def evaluate(self, x_valid, y):
        right, wrong = 0, 0
        count = 0
        y_preds = []
        for x, y in zip(x_valid, y):
            y_pred = self.model(x)
            y_preds.append(y_pred)
            if count % 500 == 0:
                print('Evaluating y_pred, y ', y_pred, round(y_pred[0]), y)
            if round(y_pred[0]) == y:
                right += 1
            else:
                wrong += 1
            count += 1

        return y_preds, right / (right + wrong)

    # Method to train the model
    def train(self, x_train, y_train, x_test, y_test, saveModel=False):
        # We use Adam optimizer with default learning rate 0.0001.
        # Change this value based on your preference
        opt = optim.Adam(self.model.parameters(), lr=0.0001)

        maxEvalRes = 0

        for e in range(100):
            print("Epoch ", e)
            for i in range(len(x_train)):
                if i % 50 == 0:
                    print('i =  ', i)
                    self.model.train()
                    opt.zero_grad()
                    y_pred, _ = self.model(x_train[i])
                    loss = loss_tracker(y_pred, torch.Tensor([y_train[i]]))
                    loss.backward()
                    opt.step()

                else:
                    self.model.train()
                    opt.zero_grad()
                    y_pred, _ = self.model(x_train[i])
                    loss = loss_tracker(y_pred, torch.Tensor([y_train[i]]))
                    loss.backward()
                    opt.step()

                # Evaluate after every 200 iterations
                if i % 200 == 0:
                    y_preds, score = self.evaluate(x_test, y_test)
                    print("Evaluate on valid set: ", score)
                    if score > maxEvalRes:
                        maxEvalRes = score
                        print("This is the best eval res, saving the model...")
                        now = datetime.now()

                        print("saving model now =", now)

                        # dd/mm/YY H:M:S
                        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
                        print("date and time =", dt_string)
                        # automatically save model after getting the best performance
                        if saveModel:
                            torch.save(self.model.state_dict(), 'my_model.pth')
                        print("just saved")

    # This method simply projects
