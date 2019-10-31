import bisect
import random
import numpy as np
from utils import mapCorrelation, image_to_world, point_in_map

class Robot(object):
    def __init__(self, N):
        p_bl = np.array([301.83 * 10 ** -3, 0])
        R_bl = np.eye(2)
        self.T_bl = np.vstack((np.hstack((R_bl, p_bl.reshape(2, 1))),np.array([[0, 0, 1]])))

        # for texture
        fsu = 585.05108211
        fsv = 585.05108211
        fst = 0
        cu = 242.94140713
        cv = 315.83800193
        K = np.array([[fsu, fst, cu], [0, fsv, cv], [0, 0, 1]])
        self.invK = np.linalg.inv(K)

        roll = 0
        pitch = 0.36
        yaw = 0.021
        R1 = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        R2 = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        R3 = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        self.R_bc = R1 @ R2 @ R3
        self.R_oc = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        self.p_bc = np.array([0.18, 0.005, 0.36]).reshape(3, 1)
        self.T_ob = np.vstack([(np.hstack((self.R_oc @ self.R_bc.T, -self.R_oc @ self.R_bc.T @ self.p_bc))),
                          np.array([[0, 0, 0, 1]])])
        self.T_bo = np.linalg.inv(self.T_ob)

        self.bp = Particle(0, 0, 0, 1)
        self.N = N
        self.particles = Particle.initial_particles(N)

    def _resample(self):
        new_particles = []
        dist = WeightedDistribution(self.particles)
        for _ in self.particles:
            p = dist.pick()
            new_particle = Particle(p.x, p.y, p.th, 1 / self.N)
            new_particles.append(new_particle)
        self.particles = new_particles

    def _find_best_particle(self):
        w_max = max(p.w for p in self.particles)
        all_sum = sum(np.exp(p.w - w_max) for p in self.particles)
        for p in self.particles:
            p.w = np.exp(p.w - w_max) / all_sum
        self.particles.sort(key = lambda x: x.w)
        self.bp = self.particles[-1]

    def moveon(self, v, w, tau, nr):
        for p in self.particles:
            vv = v + nr * np.random.normal(0, 1)
            ww = w + nr * np.random.normal(0, 1)

            p.th = p.th + ww * tau
            p.x = p.x + tau * vv * np.sinc(ww * tau / 2 / np.pi) * np.cos(p.th + ww * tau / 2)
            p.y = p.y + tau * vv * np.sinc(ww * tau / 2 / np.pi) * np.sin(p.th + ww * tau / 2)

    def update_particles(self, xlidar, ylidar, im, xmin, ymin, res):
        for p in self.particles:
            xw, yw = image_to_world(self.T_bl, xlidar, ylidar, p)
            xmap, ymap = point_in_map(xw, yw, xmin, ymin, res)

            p.w, index = mapCorrelation(im, xmap, ymap)
            p.x = p.x + res * (index[0] - 4) # change the order of index[1] and index[0] if not works well
            p.y = p.y + res * (index[1] - 4)
        self._find_best_particle()
        self._resample()



class Particle(object):
    def __init__(self, x, y, th, w):
        self.x = x
        self.y = y
        self.th = th
        self.w = w

    def __repr__(self):
        return "(%f, %f, %f, w=%f)" % (self.x, self.y, self.th, self.w)

    @classmethod
    def initial_particles(cls, count):
        return [cls(0, 0, 0, 1 / count) for _ in range(0, count)]



class WeightedDistribution(object):
    def __init__(self, state):
        accum = 0.0
        self.state = [p for p in state if p.w > 0]
        self.distribution = []
        for x in self.state:
            accum += x.w
            self.distribution.append(accum)

    def pick(self):
        try:
            return self.state[bisect.bisect_left(self.distribution, random.uniform(0, 1))]
        except IndexError:
            # Happens when all particles are improbable w=0
            return None


