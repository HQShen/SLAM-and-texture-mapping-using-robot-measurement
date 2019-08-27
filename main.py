
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm
from utils import *
from Map import Map
from Robot import Robot, Particle

dataset = 23
N = 100 # number of particles
nr = 0.8 # noise rate
texture_on = True if dataset in [20,21] else False
floor_threshold = 0.2

result_path = 'result' + str(dataset)
if not os.path.exists(result_path):
    os.mkdir(result_path)

data = data_preprocess(dataset, texture_on)
v = data['v']
w = data['w']
stamps = data['stamps']
ranges = data['ranges']
if texture_on:
    rgb = data['rgb']
    disp = data['disp']
    rgb_path = 'data/dataRGBD/RGB' + str(dataset)
    disp_path = 'data/dataRGBD/Disparity' + str(dataset)

# initialization
world = Map(-30, -30, 30, 30, 0.1, stamps.shape[0], texture = texture_on)
robot = Robot(N)
world.initialize_map(robot.T_bl, ranges[:, 0], robot.bp)

for i in tqdm.trange(stamps.shape[0] - 1):
   tau = stamps[i+1] - stamps[i]
   xlidar, ylidar = detect(ranges[:, i + 1])
   # move all particles
   robot.moveon(v[i], w[i], tau, nr)
   # update particles
   robot.update_particles(xlidar, ylidar, world.map, world.xmin, world.ymin, world.res)
   # update map
   world.update_map(robot.T_bl, xlidar, ylidar, robot.bp, i)

   if texture_on:
       rgb_fn = rgb_path + '/rgb' + str(dataset) + '_%d.png' % rgb[i]
       disp_fn = disp_path + '/disparity' + str(dataset) + '_%d.png' % disp[i]
       world.update_rgbd(disp_fn, rgb_fn, robot.invK, robot.T_bo, robot.bp, floor_threshold)

   if i % 20 == 10:
       save_path = os.path.join(result_path, 'result%d' % (int(i / 20)))
       world.show_and_save(i, robot.bp, save_path)

generate_gif(result_path, dataset, texture_on)


