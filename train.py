# import vrep
# import numpy as np
# import math
# import time
# import cv2
# from robot import Robot
# from scipy.spatial.transform import Rotation
# import random
# from utils import *

# def rotEuler2Others(euler_angle):
#     rot_euler = Rotation.from_euler('ZYX', euler_angle)
#     # q = rot_gate.as_quat()
#     quat = rot_euler.as_quat()
#     rotvec = rot_euler.as_rotvec()
#     rotmatrix = rot_euler.as_matrix()
#     # print('quat: ',quat)
#     return quat

# # new test
# reset_position = np.array([-0.2, -0.1, 0.3,  0.000000e+00, 1.000000e+00,  0.000000e+00,  6.123234e-17])
# #  self.move_to_position2()
# workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]])
# ur5_robot = Robot(workspace_limits)

# # color_img, depth_img = ur5_robot.get_kinect_data()
# # cv2.imwrite('rrr.png',color_img[36:436, 120:520, :])


# ur5_robot.set_random_positions_for_object()
# ur5_robot.move_to_position(reset_position)

# color_img, depth_img = ur5_robot.get_kinect_data()
# center1,angle1 = object_grasp_detection(color_img,depth_img)
# # print(center1)
# # print(angle1)

# for ind in range(len(center1)):
#     cent = center1[ind]
#     ang = angle1[ind]
#     pos_x, pos_y = image_coordinate_to_robot_coordinate(cent, depth_img)
#     tmp_pos = [pos_x, pos_y]
#     tmp_quat = rotEuler2Others([(90-abs(ang)), 0, -np.pi])#rotEuler2Others([-np.pi, ang, -np.pi/2])#rotEuler2Others([-np.pi/2, ang, -np.pi])#rotEuler2Others([-(90-abs(ang)),-0, -np.pi])
#     ur5_robot.move_to_position(np.array([tmp_pos[0],tmp_pos[1], 0.25, tmp_quat[0], tmp_quat[1],tmp_quat[2],tmp_quat[3]]))
#     ur5_robot.move_to_position(np.array([tmp_pos[0],tmp_pos[1], 0.06, tmp_quat[0], tmp_quat[1],tmp_quat[2],tmp_quat[3]]))
#     ur5_robot.closeRG2()
#     ur5_robot.move_to_position(np.array([tmp_pos[0],tmp_pos[1], 0.25, tmp_quat[0], tmp_quat[1],tmp_quat[2],tmp_quat[3]]))
#     ur5_robot.move_to_position(np.array([-0.5+random.uniform(-0.06,0.06),-0.5+random.uniform(-0.06,0.06), 0.3, 0.000000e+00, 1.000000e+00,  0.000000e+00,  6.123234e-17]))
#     ur5_robot.openRG2()
#     time.sleep(1)
#     print('---------------------- grasp attempt ', str(ind+1), ' ----------------------')
#     print('grasp position: [', pos_x, pos_y, '0.06]')
#     print('grasp angle: ', (90-abs(ang)))
#     # print('grasp result: ',)
#     print('\n')


#------------------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------------#

import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import time
import shutil
from arm_env import ArmEnv
import random
import utils
from tensorboardX import SummaryWriter
from prioritized_memory import Memory
import os
from models import Dqn
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#set hyper parameters
DIM_ACTIONS = 224 * 224 * 8
DIM_STATES = (224, 224)
CHANNELS = 8
MEMORY_CAPACITY = 500000
LR = 1e-4
GAMMA = 0.5
Q_NETWORK_ITERATION = 100
BATCH_SIZE = 32
EPISODES = 50#30

def main():
    env = ArmEnv()
    net = Dqn()
    # net.cuda()
    EPSILON = 0.8
    print("Start data collection, the DQN is collecting experience...")
    step_counter_list = []
    log_writer = SummaryWriter('logs/')  # ('logs/')
    log_txt_writer = open('logs/train_log.txt','a')
    step_counter = 0
    time_start = time.time()

    invalid_count = 0
    last_invalid_id = -1

    for episode in range(EPISODES):
        state, obj_num = env.reset()
        sum_reward = 0
        step_count_every_ep = 0
        success_grasp_obj_num = 0
        while True:
            step_counter += 1
            step_count_every_ep += 1
            ac_type, action = net.choose_action(state,EPSILON)
            EPSILON = EPSILON + step_counter*1e-6

            next_state, reward, done = env.step(ac_type,action,state,step_counter)
            if done == -1:
                print('Env error occured, restart simulation...')
                break

            if env.check_scene_state():
                print('Scene is empty, episode finish...')
                break

            net.store_trans(state, action, reward, next_state,done)
            sum_reward += reward

            if net.memory.tree.n_entries >= 150:#1000,1100,1100
                l, mean_q, mse_l = net.learn()
                print('step:'+str(step_counter)+', loss:'+str(float(l))+', mean_q:'+str(float(mean_q))+', mse_loss:'+str(float(mse_l)))
                if step_counter >= 160:
                    if step_counter == 160:
                        time_start = time.time()
                    log_writer.add_scalar('loss', float(l), global_step=step_counter)
                    log_writer.add_scalar('mean_q', float(mean_q), global_step=step_counter)
                    log_writer.add_scalar('mse_loss', float(mse_l), global_step=step_counter)
                    log_txt_writer.write('used time:'+str((time.time()-time_start)/60)+',step:'+str(step_counter)+',loss:'+str(float(l))+',mse_loss:'+str(float(mse_l)))
                    log_txt_writer.write('\n')

            if  net.learn_counter % 100 == 0 and net.learn_counter > 0:
                torch.save(net.eval_net.state_dict(), 'models/step_' + str(net.learn_counter) + '_params.pkl')
                print('#####################   save model   #####################')

            if done:
                success_grasp_obj_num += 1

            if success_grasp_obj_num == obj_num or step_count_every_ep == 40:
                print("episode {}, the sum reward is {}".format(episode, round(sum_reward, 4)))
                break

            state = next_state

        # if (episode+1) % 200 == 0 and step_counter >= 1200:
        #     torch.save(net.eval_net.state_dict(), 'models/ep_' + str(episode+1) + '_params.pkl')
        #     print('#####################   save model   #####################')
    torch.save(net.eval_net.state_dict(), 'models/final_params.pkl')
    log_txt_writer.close()

# def train_from_checkpoint(model_path,start_step):
#     net = Dqn()
#     net.eval_net.load_state_dict(torch.load('models/' + model_path))
#     print('load model ',model_path)
#     # net.cuda()
#     EPSILON = Train_Configs.EPSILON
#     print("Start data collection, the DQN is collecting experience...")
#     step_counter_list = []
#     log_writer = SummaryWriter('logs/')  # ('logs/')
#     log_txt_writer = open('logs/train_log.txt','a')
#     step_counter = start_step#0
#     time_start = time.time()

#     invalid_count = 0
#     last_invalid_id = -1

#     for episode in range(Train_Configs.EPISODES):
#         state, obj_num = env.reset()
#         sum_reward = 0
#         step_count_every_ep = 0
#         success_grasp_obj_num = 0
#         while True:
#             step_counter += 1
#             step_count_every_ep += 1
#             ac_type, action = net.choose_action(state,EPSILON)
#             EPSILON = Train_Configs.EPSILON + step_counter*1e-6

#             if action == -1:
#                 print('no object detected, restart a new env...')
#                 break
#                 # if last_invalid_id == -1 or step_counter - last_invalid_id <= 10:
#                 #     invalid_count += 1
#                 #     last_invalid_id = step_counter
#                 #     #####
#                 #     if invalid_count == 3:
#                 #         last_invalid_id = -1
#                 #         invalid_count = 0
#                 #         print('no object detected, restart a new env...')
#                 #         break
#                 # action = np.random.randint(0, Train_Configs.DIM_ACTIONS)
#             next_state, reward, done = env.step(ac_type,action,state,step_counter)
#             if done == -1:
#                 print('Env error occured, restart simulation...')
#                 break
#             net.store_trans(state, action, reward, next_state,done)
#             sum_reward += reward

#             if net.memory.tree.n_entries >= 1000:#1000,1100,1100
#                 l, mean_q, mask_l = net.learn()
#                 if step_counter >= 1100:
#                     if step_counter == 1100:
#                         time_start = time.time()
#                     log_writer.add_scalar('loss', float(l), global_step=step_counter)
#                     log_writer.add_scalar('mean_q', float(mean_q), global_step=step_counter)
#                     log_writer.add_scalar('mask_loss', float(mask_l), global_step=step_counter)
#                     log_txt_writer.write('used time:'+str((time.time()-time_start)/60)+',step:'+str(step_counter)+',loss:'+str(float(l))+',mean_q:'+str(float(mean_q))+',mask_loss:'+str(float(mask_l)))
#                     log_txt_writer.write('\n')

#             if  net.learn_counter % 1000 == 0 and net.learn_counter > 0:
#                 torch.save(net.eval_net.state_dict(), 'models/step_' + str(net.learn_counter + start_step) + '_params.pkl')
#                 print('#####################   save model   #####################')

#             if done == 1:
#                 success_grasp_obj_num += 1
#             if success_grasp_obj_num == obj_num or step_count_every_ep == 100:
#                 print("episode {}, the sum reward is {}".format(episode, round(sum_reward, 4)))
#                 break

#             state = next_state

#         # if (episode+1) % 200 == 0 and step_counter >= 1200:
#         #     torch.save(net.eval_net.state_dict(), 'models/ep_' + str(episode+1) + '_params.pkl')
#         #     print('#####################   save model   #####################')
#     torch.save(net.eval_net.state_dict(), 'models/final_params.pkl')
#     log_txt_writer.close()

if __name__ == '__main__':

    main()
    # train_from_checkpoint('step_115000_params.pkl',115000)
    # train_from_checkpoint('step_75000_params.pkl',75000)