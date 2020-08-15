import gym
import gym_grand_prix
import numpy as np
from math import pi

class Agent:

    def __init__(self):
        self.prev_sin = None
        self.prev_cos = None
        self.prev_steering = None
        self.prev_values_valid = False

    def optimal_action(self, sensor_info, p1=0.45, p2=1.0):
        """
        Найти оптимальное управление action = (steering, acceleration)
        как функцию данных датчиков

        """
        collision = False
        v = sensor_info[0]
        sin = -sensor_info[1]
        ladar = np.array(sensor_info[2:])
        l = len(ladar)
        n = int((l - 1) / 2)  # индекс в массиве ladar для направления вперед
        s = ladar[n]  # расстояние до стенки прямо по курсу
        # самое приоритетное действие - если близко по курсу стенка - тормозим!
        if v > pow(p1 * s, 0.5):
            action = (0, -.75)
            # print("Экстренное ториожение!")
            return action

        # определим косинус угла, зная синус и историю руления
        if self.prev_values_valid:
            cos = pow(1 - sin * sin, 0.5)  # косинус угла между вектором скорости и направлением на центр (но знак
            # неизвестен)
            if self.prev_steering == 0 and sin > self.prev_sin:  # синус вырос - значит мы приближаемся к центру,
                # косинус отрицательный
                cos *= -1.
            else:
                if sin * self.prev_sin < 0 and abs(sin) > .1:  # синус резко сменил знак - вероятно из-за отскока от
                    # стены.
                    if self.prev_cos is None:
                        cos = None
                    else:
                        cos *= -np.sign(self.prev_cos)  # значит и косинус поменял знак.
                    collision = True
                    # print("Collision detected!")
                else:
                    cos = None  #

        else:
            # делаем шаг вперед чтобы понять ориентацию (знак косинуса)
            action = (0, .75)
            # print("sin=%.3f" % (sin))
            # print("action: ", action)
            self.prev_sin = sin
            self.prev_steering = action[0]
            self.prev_values_valid = True
            return action

        fi = np.linspace(-pi / 2, pi / 2, l)  # углы между вектором скорости и направлением ладара
        if cos is None:
            dist = ladar * (sin * np.cos(fi))
        else:
            dist = ladar * (
                    sin * np.cos(fi) + cos * np.sin(fi))  # проекции расстояний ладара на "оптимальное" направление (
        # перпендикулярное направлению на центр)
        i = np.argmax(dist)  # индекс максимального значения проекции

        if i == n:  # мы уже движемся в правильном направлении - тогда либо ускоряемся, либо тормозим
            action = (0, .75) if v < pow(p1 * max(s - p2, 0), 0.5) else (0, -.75)
        else:  # есть более выгодное направление - рулим в эту сторону
            action = (1, .75) if i > n else (-1, .75)
        # print(dist, i)
        # if cos is None:
        #     print("sin=%.3f cos Unknown" % sin)
        # else:
        #     print("sin=%.3f cos=%.3f" % (sin, cos))
        # print("action: ", action)
        # input()
        self.prev_sin = sin
        self.prev_cos = cos
        self.prev_steering = action[0]
        self.prev_values_valid = True
        return action


env = gym.make('GrandPrix-v0')
done = False
vision = env.reset()
a = Agent()
while not done:
    action = a.optimal_action(vision)
    vision, reward, done, _ = env.step(action)
    env.render()
    # print(f"New state ={vision}")
    # print(f"reward={reward} , Done={done}")
env.close()