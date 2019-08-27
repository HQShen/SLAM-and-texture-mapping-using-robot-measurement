import numpy as np
import matplotlib.pyplot as plt
import os
import imageio

def data_preprocess(dataset, texture = False):
    # load data
    with np.load("data/Encoders%d.npz" % dataset) as data:
        encoder_counts = data["counts"]  # 4 x n encoder counts
        encoder_stamps = data["time_stamps"]  # encoder time stamps

    with np.load("data/Hokuyo%d.npz" % dataset) as data:
        lidar_angle_min = data["angle_min"]  # start angle of the scan [rad]
        lidar_angle_max = data["angle_max"]  # end angle of the scan [rad]
        lidar_angle_increment = data["angle_increment"]  # angular distance between measurements [rad] 0.25du
        lidar_range_min = data["range_min"]  # minimum range value [m]
        lidar_range_max = data["range_max"]  # maximum range value [m]
        lidar_ranges = data["ranges"]  # range data [m] (Note: values < range_min or > range_max should be discarded)
        lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans

    with np.load("data/Imu%d.npz" % dataset) as data:
        imu_angular_velocity = data["angular_velocity"]  # angular velocity in rad/sec
        imu_linear_acceleration = data["linear_acceleration"]  # Accelerations in gs (gravity acceleration scaling)
        imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements

    # pre-processing
    ang = imu_angular_velocity[2, :]
    FR = (encoder_counts[0, :] + encoder_counts[2, :]) / 2 * 0.0022
    FL = (encoder_counts[1, :] + encoder_counts[3, :]) / 2 * 0.0022
    Ft = (FR + FL) / 2
    # syncronization
    stamps = lidar_stamps
    vv = np.zeros(stamps.shape[0]) # linear velocity
    ww = np.zeros(stamps.shape[0]) # angular velocity
    encoder_pin = 1
    imu_pin = 1
    for i in range(1, stamps.shape[0]):
        tem_Ft = 0
        while encoder_pin < encoder_stamps.shape[0] and stamps[i] - stamps[0] > \
                encoder_stamps[encoder_pin] - encoder_stamps[0]:
            tem_Ft += Ft[encoder_pin]
            encoder_pin += 1
        vv[i - 1] = tem_Ft / (stamps[i] - stamps[i - 1])

        tem_w = 0
        num_w = 0
        while imu_pin < imu_stamps.shape[0] and stamps[i] - stamps[0] > imu_stamps[imu_pin] - imu_stamps[0]:
            tem_w += ang[imu_pin]
            imu_pin += 1
            num_w += 1
        if num_w != 0:
            ww[i - 1] = tem_w / num_w

    pro_data = {}
    pro_data['v'] = vv
    pro_data['w'] = ww
    pro_data['stamps'] = stamps
    pro_data['ranges'] = lidar_ranges

    if texture:
        with np.load("data/Kinect%d.npz" % dataset) as data:
            disp_stamps = data["disparity_time_stamps"]  # acquisition times of the disparity images
            rgb_stamps = data["rgb_time_stamps"]  # acquisition times of the rgb images

        rgb = []
        disp = []
        rgb_pin = 1
        disp_pin = 1
        rgb_num = len(os.listdir('data/dataRGBD/RGB%d' % dataset))
        disp_num = len(os.listdir('data/dataRGBD/Disparity%d' % dataset))
        for i in range(1, stamps.shape[0]):
            while rgb_pin < rgb_num and stamps[i] - stamps[0] > \
                    rgb_stamps[rgb_pin] - rgb_stamps[0]:
                rgb_pin += 1
            rgb.append(rgb_pin)

            while disp_pin < disp_num and stamps[i] - stamps[0] > \
                    disp_stamps[disp_pin] - disp_stamps[0]:
                disp_pin += 1
            disp.append(disp_pin)
        pro_data['rgb'] = rgb
        pro_data['disp'] = disp

    return pro_data


def homo(px, py):
    tem = np.ones([3, px.shape[0]])
    tem[0,:] = px
    tem[1,:] = py
    return tem

def detect(points, detect_range_max = 30, detect_range_min = 0.1):
    angles = np.arange(-135, 135.25, 0.25) * np.pi / 180.0
    indValid = np.logical_and((points < detect_range_max), (points > detect_range_min))
    points = points[indValid]
    angles = angles[indValid]
    xlidar = points * np.cos(angles)
    ylidar = points * np.sin(angles)
    return xlidar, ylidar

def wTb(particle):
    position = np.array([particle.x, particle.y])
    R = np.array([[np.cos(particle.th), -np.sin(particle.th)],
                  [np.sin(particle.th), np.cos(particle.th)]])
    T_wb = np.vstack((np.hstack((R, position.reshape(2, 1))),
                      np.array([[0, 0, 1]])))
    return T_wb

def image_to_world(T_bl, xlidar, ylidar, particle):
    T_wb = wTb(particle)
    ans = T_wb @ T_bl @ homo(xlidar, ylidar)
    xw = ans[0]
    yw = ans[1]
    return xw, yw


def point_in_map(xw, yw, xmin, ymin, res):
    xmap = np.ceil((xw - xmin) / res).astype(np.int16) - 1
    ymap = np.ceil((yw - ymin) / res).astype(np.int16) - 1
    ans = np.vstack([xmap, ymap])
    ans = np.unique(ans, axis=1)
    return ans[0], ans[1]


def mapCorrelation(im, xmap, ymap, n=9):
    nx = im.shape[0]
    ny = im.shape[1]
    cr = np.zeros((n, n))
    limit = int(np.floor(n / 2))
    range_index = list(range(-limit, limit + 1))
    for j in range(n):
        y = ymap + range_index[j]
        for i in range(n):
            x = xmap + range_index[i]
            valid = np.logical_and(np.logical_and((y >= 0), (y < ny)), \
                                   np.logical_and((x >= 0), (x < nx)))
            cr[i, j] = np.sum(im[x[valid], y[valid]])

    if cr[4, 4] != np.max(cr):
        index = np.unravel_index(cr.argmax(), cr.shape)
    else:
        index = (4, 4)
    return np.max(cr), index

def bresenham2D(sx, sy, ex, ey):
    '''
    Bresenham's ray tracing algorithm in 2D.
    Inputs:
        (sx, sy)	start point of ray
        (ex, ey)	end point of ray
    '''
    sx = int(round(sx))
    sy = int(round(sy))
    ex = int(round(ex))
    ey = int(round(ey))
    dx = abs(ex - sx)
    dy = abs(ey - sy)
    steep = abs(dy) > abs(dx)
    if steep:
        dx, dy = dy, dx  # swap

    if dy == 0:
        q = np.zeros((dx + 1, 1))
    else:
        q = np.append(0, np.greater_equal(
            np.diff(np.mod(np.arange(np.floor(dx / 2), -dy * dx + np.floor(dx / 2) - 1, -dy), dx)), 0))
    if steep:
        if sy <= ey:
            y = np.arange(sy, ey + 1)
        else:
            y = np.arange(sy, ey - 1, -1)
        if sx <= ex:
            x = sx + np.cumsum(q)
        else:
            x = sx - np.cumsum(q)
    else:
        if sx <= ex:
            x = np.arange(sx, ex + 1)
        else:
            x = np.arange(sx, ex - 1, -1)
        if sy <= ey:
            y = sy + np.cumsum(q)
        else:
            y = sy - np.cumsum(q)
    return np.vstack((x, y))


def generate_gif(path, dataset, texture_on):
    print('Generating video...')
    name = 'result_of_dataset' + str(dataset) + '_map.gif'
    gif_path = os.path.join(path, name)
    images = []
    filenames = sorted(os.listdir(path))
    for filename in filenames:
        if filename[-7:] == 'map.png':
            tem_path = os.path.join(path, filename)
            images.append(imageio.imread(tem_path))
    imageio.mimsave(gif_path, images)

    if texture_on:
        name = 'result_of_dataset' + str(dataset) + '_texture.gif'
        gif_path = os.path.join(path, name)
        images = []
        filenames = sorted(os.listdir(path))
        for filename in filenames:
            if filename[-11:] == 'texture.png':
                tem_path = os.path.join(path, filename)
                images.append(imageio.imread(tem_path))
        imageio.mimsave(gif_path, images)






