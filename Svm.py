import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
             
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

class SvmClassifier:

    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        # считаем ядра. Они нужны если у нас нелинейное разделеление, например, как у нас два круга - то тут лучше и спользовать полиномальное ядро
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        # считаем  двойственную задачу программирования - находим минимум функции. Находим как должна пройти наша кривая, чтобы быть равноудаленной ото всех
        # просто можнон исользоптьв функцию из либы для подсчета, так как самим сложно считать
        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples), 'd')
        b = cvxopt.matrix(0.0)

        # это вроде говорит, что надо ли использовать софт или не софт линию. Софт - это когда могут быть данные внутри полосы резделения
        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x']) # наши иксы, минимальные - то есть при этих иксах наша функция наиболее приближена. Это даже не иксы, а веса, при которых фукнция наиболее приближена 

        # отсекаем все, что маленькое
        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        # берем только знак от того, что получилось при предсказаниии +1 или -1
        return np.sign(self.project(X))

    # @staticmethod
    # def __calculate_kernel(distance: float, kernel_type: KernelType):
    #     r = distance
    #     # оптимальное Епанчиклва
    #     if kernel_type == KernelType.E:
    #         return 3/4*(1-r**2)
    #     # квартическое
    #     if kernel_type == KernelType.Q:
    #         return 15/16*(1-r**2)**2
    #     # теругольное
    #     if kernel_type == KernelType.T:
    #         return 1-abs(r)
    #     # гуасовское
    #     if kernel_type == KernelType.G:
    #         return pow(2 * pi, (-0.5))*exp(-0.5*r**2)
    #     # прямоугольное rectangle
    #     if kernel_type == KernelType.R:
    #         return 1/2
    #     return 1