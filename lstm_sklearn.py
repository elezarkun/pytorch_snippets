import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim


def make_data(N):
    x_test = np.random.randint(200, size=(N, 5)).astype(np.float)
    x_train = np.random.randint(200, size=(N, 5)).astype(np.float)
    print(x_train.shape)
    print(np.sum(x_train, axis=1))
    y_train = (np.sum(x_train, axis=1) > 100 * 5).astype(np.float32)
    print(y_train.shape)
    print(type(y_train))
    print(type(y_train[0]))
    y_train = y_train.reshape(-1, 1)
    y_test = (np.sum(x_test, axis=1) > 100 * 5).astype(np.float32)
    y_test = y_test.reshape(-1, 1)
    # print(y_test[:10])
    # y_train = y_train.squeeze()
    # x_train = torch.from_numpy(x_train)
    # x_test = torch.from_numpy(x_test)
    # y_train = torch.from_numpy(y_train).long()
    print x_train.shape, y_train.shape, x_test.shape, y_test.shape
    print y_train
    return x_train, y_train, x_test, y_test


class PDLSTM(nn.Module):
    def __init__(self, lr=1.0, verbose=True):
        super(PDLSTM, self).__init__()
        self.lr = lr
        self.loss = nn.CrossEntropyLoss(size_average=True)
        self.verbose = verbose
        print 'initialized'

    def build_model(self, input_dim, output_dim):
        self.embedding = nn.Embedding(775, 100)
        self.lstm = nn.LSTM(100, 20, num_layers=1, batch_first=True)
        self.fc = nn.Linear(20, output_dim)
        self.ReLU = nn.ReLU()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        out1 = self.embedding(x)
        out2, _ = self.lstm(out1)
        out3 = self.fc(out2[:, -1, :])
        return out3.squeeze()

    def train(self, x_val, y_val):
        x = Variable(x_val, requires_grad=False)
        y = Variable(y_val, requires_grad=False)
        # Reset gradient
        self.optimizer.zero_grad()

        # Forward
        fx = self.forward(x)
        # print fx
        # print '111',fx.shape, y.shape
        output = self.loss.forward(fx, y.squeeze())
        self.cost = output.data[0]

        # Backward
        output.backward()

        # Update parameters
        self.optimizer.step()

    def predict(self, x_val):
        return self.predict_proba(x_val).argmax(axis=1)

    def predict_proba(self, x_val):
        x = Variable(x_val, requires_grad=False)
        output = self.forward(x)
        return output.data.numpy()

    def sigmoid(x):
        return 1 / (1 + np.exp(-1.0 * x))

    def fit(self, X, y, eval_set=None, epoch=100):
        torch.manual_seed(42)
        trX = torch.from_numpy(X).long()
        trY = torch.from_numpy(y).long()
        if eval_set is not None:
            X_eval, teY = eval_set
            teX = torch.from_numpy(X_eval).long()

        batch_size = len(y)
        self.build_model(X.shape[1], 2)
        for i in range(0, int(epoch)):
            self.cost = 0
            num_batches = len(y) // batch_size
            for k in range(num_batches):
                start, end = k * batch_size, (k + 1) * batch_size
                self.train(trX[start:end], trY[start:end])

            if self.verbose & i % 1000 == 0:
                if eval_set is not None:
                    evalX = teX
                    evalY = teY
                else:
                    evalX = trX
                    evalY = trY

                accuracy = 100. * accuracy_score(evalY[:, 0],
                                                 self.predict(evalX))
                accuracy_train = 100. * accuracy_score(trY[:, 0],
                                                       self.predict(trX))
                AUC = roc_auc_score(evalY, self.predict_proba(evalX)[:, 1])
                AUC_train = roc_auc_score(trY, self.predict_proba(trX)[:, 1])
                print('Epoch AUC: %f, AUC_train: %f' % (AUC, AUC_train))
                # print(predict(model, teX)[:10])
                print("Epoch %d, cost = %f, acc = %.2f%%, acc_train = %.2f%%"
                      % (i, self.cost / num_batches, accuracy, accuracy_train))

    def fit_further(self, X, y, eval_set=None, epoch=100):
        torch.manual_seed(42)
        trX = torch.from_numpy(X).long()
        trY = torch.from_numpy(y).long()
        if eval_set is not None:
            X_eval, teY = eval_set
            teX = torch.from_numpy(X_eval).long()
        batch_size = len(y)
        for i in range(0, int(epoch)):
            self.cost = 0
            num_batches = len(y) // batch_size
            for k in range(num_batches):
                start, end = k * batch_size, (k + 1) * batch_size
                self.train(trX[start:end], trY[start:end])

            if self.verbose & i % 1000 == 0:
                if eval_set is not None:
                    evalX = teX
                    evalY = teY
                else:
                    evalX = trX
                    evalY = trY

                accuracy = 100. * accuracy_score(evalY[:, 0],
                                                 self.predict(evalX))
                accuracy_train = 100. * accuracy_score(trY[:, 0],
                                                       self.predict(trX))
                AUC = roc_auc_score(evalY, self.predict_proba(evalX)[:, 1])
                AUC_train = roc_auc_score(trY, self.predict_proba(trX)[:, 1])
                print('Epoch AUC: %f, AUC_train: %f' % (AUC, AUC_train))
                # print(predict(model, teX)[:10])
                print("Epoch %d, cost = %f, acc = %.2f%%, acc_train = %.2f%%"
                      % (i, self.cost / num_batches, accuracy, accuracy_train))


if __name__ == "__main__":
    trX, trY, teX, teY = make_data(500000)
    print np.mean(trY), np.mean(teY)
    clf = PDLSTM(lr=1e-1)
    clf.fit(trX, trY, eval_set=(teX, teY), epoch=10)
    clf.lr = 1e-3
    clf.fit_further(trX, trY, eval_set=(teX, teY), epoch=40)
    clf.lr = 1e-4
    clf.fit_further(trX, trY, eval_set=(teX, teY), epoch=10)
    # clf.lr = 1e-5
    # clf.fit_further(trX, trY, eval_set=(teX, teY), epoch=80)
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression()
    clf.fit(trX, trY)
    print roc_auc_score(teY, clf.predict_proba(teX)[:, 1])
    print accuracy_score(teY, clf.predict(teX))
