import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
from torch.autograd import Variable
from torch import optim


def make_data(N):
    N = 1
    x_test = (np.random.random((100, N))).astype(np.float32).reshape(-1, N)
    x_train = (np.random.random((100, N))).astype(np.float32).reshape(-1, N)
    add_train = (np.random.random((100, N))).astype(np.float32).reshape(-1, N)
    add_test = (np.random.random((100, N))).astype(np.float32).reshape(-1, N)
    y_train = (np.sum(x_train, axis=1).reshape(-1, 1) +
               (add_train - 0.5) > 0.5).astype(np.float32)
    y_train = y_train.reshape(-1, 1)
    y_test = (np.sum(x_test, axis=1).reshape(-1, 1) +
              (add_test - 0.5) > 0.5).astype(np.float32).reshape(-1, 1)
    # print(y_test[:10])
    y_train = y_train.squeeze()
    # x_train = torch.from_numpy(x_train)
    # x_test = torch.from_numpy(x_test)
    # y_train = torch.from_numpy(y_train).long()
    return x_train, y_train, x_test, y_test


class LogisticRegression():
    def __init__(self, lr=1.0, verbose=True):
        self.lr = lr
        self.loss = torch.nn.CrossEntropyLoss(size_average=True)
        self.verbose = verbose

    def build_model(self, input_dim, output_dim):
        self.model = torch.nn.Sequential()
        self.model.add_module("linear",
                              torch.nn.Linear(input_dim, output_dim))
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

    def train(self, x_val, y_val):
        x = Variable(x_val, requires_grad=False)
        y = Variable(y_val, requires_grad=False).squeeze()

        # Reset gradient
        self.optimizer.zero_grad()

        # Forward
        fx = self.model.forward(x)
        output = self.loss.forward(fx, y)

        # Backward
        output.backward()
        self.cost_epoch += output.data[0]

        # Update parameters
        self.optimizer.step()

    def predict(self, x_val):
        return self.predict_proba(x_val).argmax(axis=1)

    def predict_proba(self, x_val):
        x = Variable(x_val, requires_grad=False)
        output = self.model.forward(x)
        return output.data.numpy()

    def sigmoid(x):
        return 1 / (1 + np.exp(-1.0 * x))

    def fit(self, X, y, eval_set=None, epoch=100):
        torch.manual_seed(42)
        trX = torch.from_numpy(X).float()
        trY = torch.from_numpy(y).long()
        if eval_set is not None:
            X_eval, teY = eval_set
            teX = torch.from_numpy(X_eval).float()

        batch_size = len(y)
        self.build_model(X.shape[1], 2)
        for i in range(0, int(epoch)):
            self.cost_epoch = 0
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
                AUC = roc_auc_score(evalY, self.predict_proba(evalX)[:, 1])
                print('Epoch AUC: %f' % AUC)
                # print(predict(model, teX)[:10])
                print("Epoch %d, cost = %f, acc = %.2f%%"
                      % (i, self.cost_epoch / num_batches, accuracy))


if __name__ == "__main__":
    trX, trY, teX, teY = make_data(1)
    clf = LogisticRegression()
    clf.fit(trX, trY, eval_set=(teX, teY))
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression()
    clf.fit(trX, trY)
    print roc_auc_score(teY, clf.predict_proba(teX)[:, 1])
    print accuracy_score(teY, clf.predict(teX))
