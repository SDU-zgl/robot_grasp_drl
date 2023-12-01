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
import os
from models import Dqn
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# #set hyper parameters
# DIM_ACTIONS = 224 * 224 * 8
# DIM_STATES = (224, 224)
# CHANNELS = 8
# MEMORY_CAPACITY = 500000
# LR = 1e-4
# GAMMA = 0.5
# Q_NETWORK_ITERATION = 100
# BATCH_SIZE = 32
# EPISODES = 100000

def evaluation(model_path):
    env = ArmEnv()
    net = Dqn()
    net.eval_net.load_state_dict(torch.load('models/'+model_path))# (torch.load('models/model_D12_M5/step_35000_params.pkl'))

    # EVALUATION_EP = 50
    MAX_STEP_FOR_EVERY_EP = 20
    for episode in range(5):
        print ('-----------------  Test episode ', str(episode + 1), ' start...  ------------------')
        state, obj_num = env.reset_eval()
        step_count = 0
        while step_count < MAX_STEP_FOR_EVERY_EP:
            if env.check_scene_state():
                print('Scene is empty, episode finish...')
                break
            action = net.choose_action_for_eval(state)
            next_state, done = env.step_eval(action,state,step_count)
            state = next_state
            step_count += 1


if __name__ == '__main__':

    evaluation('step_800_params.pkl')