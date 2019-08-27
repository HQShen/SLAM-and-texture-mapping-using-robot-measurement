import numpy as np
import matplotlib.pyplot as plt
from utils import *
from PIL import Image
import cv2

class Map(object):
    def __init__(self, xmin, ymin, xmax, ymax, res, dist, texture = False):
        self.res = res  # meters
        self.xmin = xmin  # meters
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.sizex = int(np.ceil((self.xmax - self.xmin) / self.res + 1))  # cells
        self.sizey = int(np.ceil((self.ymax - self.ymin) / self.res + 1))
        self.map = np.zeros((self.sizex, self.sizey), dtype=np.int8)  # DATA TYPE: char or int8
        self.value = np.zeros((self.sizex, self.sizey), dtype=np.float64)
        self.show = np.zeros((self.sizex, self.sizey), dtype=np.int8)
        self.rgbd = np.zeros((self.sizex, self.sizey, 3), dtype=np.float64)
        self.rgbd_w = np.zeros((self.sizex, self.sizey), dtype=np.int16)
        self.route = np.zeros([2, dist])
        self.trust = 0.8
        self.log_t = np.log(self.trust / (1 - self.trust))
        self.texture = texture

    def initialize_map(self, T_bl, points, bp):
        xlidar, ylidar = detect(points)
        xw, yw = image_to_world(T_bl, xlidar, ylidar, bp)
        xmap, ymap = point_in_map(xw, yw, self.xmin, self.ymin, self.res)
        self.route[0, 0], self.route[1, 0] = point_in_map(0, 0, self.xmin, self.ymin, self.res)
        self._change_map(self.route[0, 0], self.route[1, 0], xmap, ymap)

    def update_map(self, T_bl, xlidar, ylidar, bp, i):
        xw, yw = image_to_world(T_bl, xlidar, ylidar, bp)
        xmap, ymap = point_in_map(xw, yw, self.xmin, self.ymin, self.res)
        self.route[0, i], self.route[1, i] = point_in_map(bp.x, bp.y, self.xmin, self.ymin, self.res)
        self._change_map(self.route[0, i], self.route[1, i], xmap, ymap)

    def _change_map(self, origin_x, origin_y, xmap, ymap):
        points = np.empty([2,0])
        for k in range(xmap.shape[0]):
            pts = bresenham2D(origin_x, origin_y, xmap[k], ymap[k])
            points = np.concatenate((points, pts), axis=1)
        points = points.astype(np.int32)
        self.value[points[0], points[1]] -= self.log_t
        self.value[xmap, ymap] += 2 * self.log_t
        self.map[self.value > 10 * self.log_t] = 1  # 0.499
        self.map[self.value < 0] = 0
        self.show[self.value > 5 * self.log_t] = 1
        self.show[self.value < -5 * self.log_t] = -1


    def update_rgbd(self, disp_fn, rgb_fn, invK, T_bo, bp, threshold = 0.2):
        fig_rgb = plt.imread(rgb_fn)
        disp_img = np.array(Image.open(disp_fn))

        dd = -0.00304 * disp_img + 3.31
        depth = 1.03 / dd
        v, u = depth.shape
        uu, vv = np.meshgrid(np.arange(u), np.arange(v))
        image_index = np.vstack([vv.reshape(-1), uu.reshape(-1)]).astype(int)
        index = np.argwhere(depth > 0)
        depth = depth.reshape(-1)
        valid_depth = depth > 0
        op = invK @ homo(index[:, 1].reshape(-1), index[:, 0].reshape(-1)) * depth[valid_depth]

        T_wb = np.zeros([4, 4])
        T_wb[0:3, 0:3] = np.array([[np.cos(bp.th), -np.sin(bp.th), 0], [np.sin(bp.th), np.cos(bp.th), 0], [0, 0, 1]])
        T_wb[:, 3] = np.array([bp.x, bp.y, 0.127, 1])
        new_op = np.vstack((op, np.ones((1, op.shape[1]))))
        world_frame = T_wb @ T_bo @ new_op
        wz = world_frame[2, :]
        valid_index = np.logical_and(wz < threshold, wz > 0)
        world_xy = world_frame[:2, valid_index]
        if world_xy.size == 0:
            return
        tx = np.ceil((world_xy[0, :] - self.xmin) / self.res).astype(np.int16) - 1
        ty = np.ceil((world_xy[1, :] - self.ymin) / self.res).astype(np.int16) - 1
        world_index, indices = np.unique(np.vstack([tx, ty]), axis=1, return_index = True)
        depth_index = image_index[:, valid_depth][:, valid_index][:, indices]

        j, i = depth_index[0], depth_index[1]
        rgb_i = np.round((i * 526.37 - dd[j, i] * (-4.5 * 1750.46) + 19276) / 585.051).astype(int)
        rgb_j = np.round((j * 526.37 + 16662.0) / 585.051).astype(int)
        tem = self.rgbd_w[world_index[1, :], world_index[0, :]]
        self.rgbd[world_index[1, :], world_index[0, :], :] = ((fig_rgb[rgb_j, rgb_i, :].T +
                                                       tem * self.rgbd[world_index[1, :], world_index[0, :], :].T)/(tem + 1)
                                                              * (tem > 5)).T
        self.rgbd_w[world_index[1, :], world_index[0, :]] += 1

    def show_and_save(self, i, bp, save_path):
        plt.figure()
        plt.imshow(np.transpose(self.show))
        plt.plot(self.route[0, :i], self.route[1, :i], 'r--', markersize=0.5)
        plt.plot(self.route[0, i], self.route[1, i], 'r', marker=(3, 0, bp.th / np.pi * 180 - 90), markersize=6)
        plt.savefig(save_path + '_map', dpi=100)
        plt.close()
        if self.texture:
            plt.figure()
            plt.imshow(self.rgbd.reshape(self.sizey, self.sizex, 3))
            plt.plot(self.route[0, :i], self.route[1, :i], 'r--', markersize=0.2)
            plt.plot(self.route[0, i], self.route[1, i], 'r', marker=(3, 0, bp.th / np.pi * 180 - 90),
                          markersize=6)
            plt.savefig(save_path + '_texture', dpi=100)
            plt.close()