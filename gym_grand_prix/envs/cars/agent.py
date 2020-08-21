import random
from abc import ABCMeta, abstractmethod
from collections import deque

import numpy as np

from gym_grand_prix.envs.cars.utils import Action
from gym_grand_prix.envs.learning_algorithms.network import Network


class Agent(metaclass=ABCMeta):
    @property
    @abstractmethod
    def rays(self):
        pass

    @abstractmethod
    def choose_action(self, sensor_info):
        pass

    @abstractmethod
    def receive_feedback(self, reward):
        pass


class SimpleCarAgent(Agent):
    def __init__(self, history_data=int(500000), nrays=5):
        """
        Создаёт машинку
        :param history_data: количество хранимых нами данных о результатах предыдущих шагов
        """
        self.evaluate_mode = False  # этот агент учится или экзаменутеся? если учится, то False
        self._rays = nrays  # выберите число лучей ладара; например, 5
        # here +2 is for 2 inputs from elements of Action that we are trying to predict
        self.neural_net = Network([self.rays + 4,
                                   100,  # внутренние слои сети: выберите, сколько и в каком соотношении вам нужно
                                   10,  # например, (self.rays + 4) * 2 или просто число
                                   1],
                                  output_function=lambda x: x, output_derivative=lambda x: 1)
        self.sensor_data_history = deque([], maxlen=history_data)
        self.chosen_actions_history = deque([], maxlen=history_data)
        self.reward_history = deque([], maxlen=history_data)
        # self.reward_history.append(0.)  # workaround
        self.step = 0
        self.avg_reward = 0.    # средняя награда за последнюю тысячу шагов
        self.sum_reward = 0.    # сумма всех наград

    @classmethod
    def from_weights(cls, layers, weights, biases):
        """
        Создание агента по параметрам его нейронной сети. Разбираться не обязательно.
        """
        agent = SimpleCarAgent()
        agent._rays = weights[0].shape[1] - 4
        nn = Network(layers, output_function=lambda x: x, output_derivative=lambda x: 1)

        if len(weights) != len(nn.weights):
            raise AssertionError("You provided %d weight matrices instead of %d" % (len(weights), len(nn.weights)))
        for i, (w, right_w) in enumerate(zip(weights, nn.weights)):
            if w.shape != right_w.shape:
                raise AssertionError("weights[%d].shape = %s instead of %s" % (i, w.shape, right_w.shape))
        nn.weights = weights

        if len(biases) != len(nn.biases):
            raise AssertionError("You provided %d bias vectors instead of %d" % (len(weights), len(nn.weights)))
        for i, (b, right_b) in enumerate(zip(biases, nn.biases)):
            if b.shape != right_b.shape:
                raise AssertionError("biases[%d].shape = %s instead of %s" % (i, b.shape, right_b.shape))
        nn.biases = biases

        agent.neural_net = nn

        return agent

    @classmethod
    def from_string(cls, s):
        from numpy import array  # это важный импорт, без него не пройдёт нормально eval
        layers, weights, biases = eval(s.replace("\n", ""), locals())
        return cls.from_weights(layers, weights, biases)

    @classmethod
    def from_file(cls, filename):
        c = open(filename, "r").read()
        return cls.from_string(c)

    def show_weights(self):
        params = self.neural_net.sizes, self.neural_net.weights, self.neural_net.biases
        #np.set_printoptions(threshold=np.nan)
        np.set_printoptions(threshold=1e9)
        return repr(params)

    def to_file(self, filename):
        c = self.show_weights()
        f = open(filename, "w")
        f.write(c)
        f.close()

    @property
    def rays(self):
        return self._rays

    def choose_action(self, sensor_info):
        # хотим предсказать награду за все действия, доступные из текущего состояния
        rewards_to_controls_map = {}
        # дискретизируем множество значений, так как все возможные мы точно предсказать не сможем
        all_actions = ((0., 0.), (0., .75), (0., -.75), (1., .75), (-1., .75))
        probabilities = np.zeros(5)
        s = 0.  # sum for softmax
        ind = 0
        for steering, acceleration in all_actions:
            action = Action(steering, acceleration)
            agent_vector_representation = np.append(sensor_info, action)
            agent_vector_representation = agent_vector_representation.flatten()[:, np.newaxis]
            predicted_reward = float(self.neural_net.feedforward(agent_vector_representation))
            probabilities[ind] = np.exp(predicted_reward)
            s += probabilities[ind]
            ind += 1
        probabilities /= s
        ind = np.random.choice(5, size=1, p=probabilities)[0]
        steering, acceleration = all_actions[ind]
        best_action = Action(steering, acceleration)
        # запомним всё, что только можно: мы хотим учиться на своих ошибках
        self.sensor_data_history.append(sensor_info)
        self.chosen_actions_history.append(best_action)
        self.reward_history.append(0.0)  # мы пока не знаем, какая будет награда, это
        # откроется при вызове метода receive_feedback внешним миром

        return best_action

    def receive_feedback(self, reward, train_every=100, reward_depth=10):
        """
        Получить реакцию на последнее решение, принятое сетью, и проанализировать его
        :param reward: оценка внешним миром наших действий
        :param train_every: сколько нужно собрать наблюдений, прежде чем запустить обучение на несколько эпох
        :param reward_depth: на какую глубину по времени распространяется полученная награда
        """
        # считаем время жизни сети; помогает отмерять интервалы обучения
        self.step += 1
        q = .001 if self.step > 1000 else 1./float(self.step)
        self.avg_reward = (1. - q)*self.avg_reward + q*reward
        self.sum_reward += reward

        # начиная с полной полученной истинной награды,
        # размажем её по предыдущим наблюдениям
        # чем дальше каждый раз домножая её на 1/2
        # (если мы врезались в стену - разумно наказывать не только последнее
        # действие, но и предшествующие)
        i = -1
        while len(self.reward_history) > abs(i) and abs(i) < reward_depth:
            self.reward_history[i] += reward
            reward *= 0.9
            i -= 1

        # Если у нас накопилось хоть чуть-чуть данных, давайте потренируем нейросеть
        # прежде чем собирать новые данные
        # (проверьте, что вы в принципе храните достаточно данных (параметр `history_data` в `__init__`),
        # чтобы условие len(self.reward_history) >= train_every выполнялось
        if not self.evaluate_mode and (len(self.reward_history) >= train_every) and not (self.step % train_every):
            X_train = np.concatenate([self.sensor_data_history, self.chosen_actions_history], axis=1)[-10*train_every:]
            y_train = np.array(self.reward_history)[-10*train_every:]
            train_data = [(x[:, np.newaxis], y) for x, y in zip(X_train, y_train)]
            self.neural_net.SGD(training_data=train_data, epochs=15, mini_batch_size=train_every, eta=0.05)
