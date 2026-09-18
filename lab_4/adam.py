import numpy as np

class Adam:
    def __init__(self, parameters, lr=1e-3, beta_1=0.9, beta_2=0.999, eps=1e-8):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

        self.parameters = parameters

        self.m = [np.zeros_like(param) for param in parameters]
        self.v = [np.zeros_like(param) for param in parameters]

        # номер шага (для bias correction)
        self.t = 1

    def step(self, grads):
        for i in range(len(self.parameters)):
            # momentum: m_t
            self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * grads[i]

            # RMSProp: v_t
            self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * grads[i] ** 2

            # bias correction m_t
            m_hat = self.m[i] / (1 - self.beta_1 ** self.t)

            # обновляем веса
            v_hat = self.v[i] / (1 - self.beta_2 ** self.t)

            self.parameters[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            self.t += 1