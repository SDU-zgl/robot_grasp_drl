"""
Environment for Robot Arm in V-REP.
"""
from robot import Robot
import numpy as np
import matplotlib.pyplot as plt
import time
from threading import Timer
import argparse
from PIL import Image
import cv2
import vrep
import utils
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

class ArmEnv(object):
    # dt = .1  # refresh rate
    # arm_lr = 100
    # grab_counter = 0

    def __init__(self):
        # Initialize camera and robot
        workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]])
        self.robot = Robot(workspace_limits)
        self.object_name_list = ['object1','object2','object3','object4','object5','object6']
        self.task_count = 0

    # def printForce(self):
    #     self.robot.record_force()
    #     t = Timer(1, self.printForce, )
    #     t.start()

    def step(self,ac_type, action_id,current_depth_path,step):

        action = utils.predict_actionID_to_execution_action(action_id,current_depth_path)
        self.robot.openRG2()
        self.robot.move_to_position([action[0],action[1],action[2]+0.2,action[3],action[4],action[5],action[6]])
        self.robot.move_to_position([action[0],action[1],0.06,action[3],action[4],action[5],action[6]])
        self.robot.closeRG2()
        self.robot.move_to_position([action[0],action[1],action[2]+0.2,action[3],action[4],action[5],action[6]])
        finish = self.robot.check_grasp_success()
        # self.robot.go_to_reset_position()
        if finish:
            return_r = 1
        else:
            return_r = -0.05

        path_depth = self.robot.get_current_state('train')
        print('step:',step,', action_type:',ac_type,', action_id:',action_id,', execute action:', np.round(action,4),'reward:',return_r)
        return path_depth,return_r,finish #new_state, return_r,finish

    def step_eval(self, action_id,current_depth_path,step):
        action = utils.predict_actionID_to_execution_action(action_id,current_depth_path)
        self.robot.openRG2()
        self.robot.move_to_position([action[0],action[1],action[2]+0.2,action[3],action[4],action[5],action[6]])
        self.robot.move_to_position([action[0],action[1],0.06,action[3],action[4],action[5],action[6]])
        self.robot.closeRG2()
        self.robot.move_to_position([action[0],action[1],action[2]+0.2,action[3],action[4],action[5],action[6]])
        finish = self.robot.check_grasp_success()

        path_depth = self.robot.get_current_state('test')
        print('step:',step, ', action_id:',action_id,', execute action:', np.round(action,4),'grasp success:',finish)
        return path_depth,finish #new_state, return_r,finish


    #----------------------------------------------------------------
    def set_random_positions_for_object(self):
        rand_pos_list = self.generate_random_position()
        rand_pos_inds = random.sample(list(range(6)),6)
        # for obj_name in self.object_name_list:
        for i in range(len(self.object_name_list)):
            obj_name = self.object_name_list[i]
            sim_ret, object_handle = vrep.simxGetObjectHandle(self.sim_client, obj_name, vrep.simx_opmode_blocking)
            rand_pos = rand_pos_list[rand_pos_inds[i]]
            drop_x = rand_pos[0]#random.uniform(self.workspace_limits[0][0] + 0.035, self.workspace_limits[0][1] - 0.035)
            drop_y = rand_pos[1]#random.uniform(self.workspace_limits[1][0] + 0.035, self.workspace_limits[1][1] - 0.035)
            object_position = [drop_x, drop_y, 0.15]
            # nishizhen positive; shunshizhen negtive
            rand_angle = np.random.randint(-60, 60)
            sim_ret, object_orientation = vrep.simxGetObjectOrientation(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            object_orientation[2] = rand_angle/180.0*np.pi#[-np.pi/2, 0, rand_angle/180.0*np.pi]
            vrep.simxSetObjectPosition(self.sim_client, object_handle, -1, object_position, vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, object_handle, -1, object_orientation, vrep.simx_opmode_blocking)
            time.sleep(0.5)

        self.task_count += 1

    def generate_random_position(self):
        random_pos_list_1 = [[-0.6493, -0.112],[-0.5, -0.112],[-0.35066, -0.112],[-0.6493, 0.112],[-0.5, 0.112],[-0.35066, 0.112]]
        random_pos_list_2 = [[-0.612, -0.1493],[-0.612, 0],[-0.612, 0.1493],[-0.388, -0.1493],[-0.388, 0],[-0.388, 0.1493]]
        if self.task_count % 2 == 0:
            pos_list = random_pos_list_1
        else:
            pos_list = random_pos_list_2

        pos1 = [pos_list[0][0]+random.uniform(-0.05,0.05), pos_list[0][1]+random.uniform(-0.05,0.05)]
        pos2 = [pos_list[1][0]+random.uniform(-0.05,0.05), pos_list[1][1]+random.uniform(-0.05,0.05)]
        pos3 = [pos_list[2][0]+random.uniform(-0.05,0.05), pos_list[2][1]+random.uniform(-0.05,0.05)]
        pos4 = [pos_list[3][0]+random.uniform(-0.05,0.05), pos_list[3][1]+random.uniform(-0.05,0.05)]
        pos5 = [pos_list[4][0]+random.uniform(-0.05,0.05), pos_list[4][1]+random.uniform(-0.05,0.05)]
        pos6 = [pos_list[5][0]+random.uniform(-0.05,0.05), pos_list[5][1]+random.uniform(-0.05,0.05)]

        return [pos1, pos2, pos3, pos4, pos5, pos6]


    # def step_eval(self, action_id,current_depth_path,step):
    #     action = my_utils.predict_actionID_to_execution_action(action_id,current_depth_path)

    #     return_r,finish = self.robot.step(action)
    #     if finish == -1:
    #         return 'null',0,-1
    #     else:
    #         # path_color,path_depth = self.robot.get_current_state()
    #         path_depth = self.robot.get_current_state_eval()
    #         if return_r == 0:
    #             return_r = return_r - 0.1
    #         print('step:',step,', action_type: 0, action_id:',action_id,', execute action:', action,'reward:',return_r)
    #         return path_depth,return_r,finish #new_state, return_r,finish

    def reset(self):
        self.robot.set_random_positions_for_object()
        path_depth = self.robot.get_current_state('train')
        return path_depth, 6

    def reset_eval(self):
        self.robot.set_random_positions_for_object()
        path_depth = self.robot.get_current_state('test')
        return path_depth, 6

    def check_scene_state(self):
        return self.robot.check_scene_empty()
    # def reset_eval(self):
    #     # self.grab_counter = 0

    #     self.robot.restart_sim()
    #     obj_nums = self.robot.add_objects(self.texture_dir)
    #     #self.robot.random_position()
    #     path_depth = self.robot.get_current_state_eval()
    #     return obj_nums,path_depth #new_state#self.robot.get_current_state()
    # def sample_action(self):
    #     # position_limits np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])
    #     # angle_limits (0,2*np.pi)
    #     s_x = random.uniform(-0.724, -0.276)
    #     s_y = random.uniform(-0.224, 0.224)
    #     s_z = random.uniform(-0.0001, 0.4)
    #     s_o = random.uniform(0, 2*np.pi)
    #     #C = 2  # 随机数的精度round(数值，精度)

    #     p_x = round(s_x, 3)
    #     p_y = round(s_y, 3)
    #     p_z = round(s_z, 4)
    #     o_z = round(s_o, 4)

    #     return (p_x,p_y,p_z,o_z)

    # def cal_change_for_R(self,p0,o0,p1,o1):
    #     num = len(p0)
    #     sum_p = 0
    #     sum_o = 0
    #     for i in range(num):
    #         sum_p = sum_p + abs(p0[i][0] - p1[i][0]) + abs(p0[i][1] - p1[i][1]) + abs(p0[i][2] - p1[i][2])
    #         sum_o = sum_o + abs(o0[i][0] - o1[i][0]) + abs(o0[i][1] - o1[i][1]) + abs(o0[i][2] - o1[i][2])

    #     sum_p /= (num*3)
    #     sum_o /= (num*3)
    #     if sum_p >= 0.01 or sum_o >= 0.01:
    #         return (sum_p - 0.01) * 0.5 + (sum_o - 0.01) * 0.5
    #     else:
    #         return 0

    # def go_to_position(self,position):
    #     self.robot.go_to_position(position)

    # def get_objects_positions(self):
    #     return self.robot.get_obj_positions_and_orientations()