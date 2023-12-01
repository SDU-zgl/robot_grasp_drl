# ------------------------------------------------------------------------------------------------------------
# Deep Q-Learning Algorithm for learning grasp model
# Input: RGBD
# ------------------------------------------------------------------------------------------------------------

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
import random
import utils
import shutil
from tensorboardX import SummaryWriter
from prioritized_memory import Memory
import os
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

#--------------------------------------
# Build network for q-value prediction
# Input: 0-1 binary mask(copy to 3 channels), dim:[3,224,224]
# Output: [8,224,224]
#--------------------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.base_model = torchvision.models.densenet121(pretrained=True)
        self.get_feature = self.base_model.features  # shape:([batch_size, 1024, 7, 7]),input[batch_size,3,224,224]

        self.conv_feat = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, CHANNELS, kernel_size=1, stride=1, bias=False)
        )#out:[batch_size,CHANNELS,7,7]

        # Lateral convolutional layer
        self.lateral_layer = nn.Conv2d(3, CHANNELS, kernel_size=1, stride=1, padding=0)# 512
        # Bilinear Upsampling
        self.up = nn.Upsample(scale_factor=32, mode='bilinear')

    def forward(self, x):
        #--- non mask based ---#
        # out1 = self.get_feature(x) # out:[batch_size,1024,7,7]
        # out2 = self.conv_feat(out1) # out:[batch_size,CHANNELS,7,7]
        # out_up =  self.up(out2) + self.lateral_layer(x)
        # return out_up#dim:[batch_size,CHANNELS,224,224]

        #--- mask based ---#
        out1 = self.get_feature(x) # out:[batch_size,1024,7,7]
        out2 = self.conv_feat(out1) # out:[batch_size,CHANNELS,7,7]
        out_up =  self.up(out2) + self.lateral_layer(x)
        # mask based modulation
        single_input_mask = x[:,:1,:,:]
        multiple_input_mask = torch.cat((single_input_mask,single_input_mask,single_input_mask,single_input_mask,single_input_mask,single_input_mask,single_input_mask,single_input_mask), dim=1)

        return out_up.mul(multiple_input_mask) #dim:[batch_size,CHANNELS,224,224]
        

class Dqn():
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.eval_net.cuda()
        self.target_net.cuda()

        # create prioritized replay memory using SumTree
        self.memory = Memory(MEMORY_CAPACITY)
        self.learn_counter = 0
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=LR,betas=(0.9, 0.99), eps=1e-08, weight_decay=2e-5)
        self.loss = nn.MSELoss(reduce=False, size_average=False)
        # self.loss = nn.SmoothL1Loss(reduce=False)  # Huber loss

        self.fig, self.ax = plt.subplots()
        self.discount_factor = GAMMA

        # ---- filter stuck action, same action and same state -----
        self.LAST_ACTION_ID = -1
        self.SAME_ACTION_COUNT = 0

        # save successful samples
        self.success_sample_action = []
        self.success_sample_state = []

    def store_trans(self, state_path, action, reward, next_state_path,done):
        ## action type: id
        trans = state_path+'#'+str(action)+'#'+str(reward)+'#'+next_state_path#np.hstack((state, [action], [reward], next_state))
        #------ calculate TD errors from (s,a,r,s')
        state = utils.load_depth_mask_heightmap_data(state_path).reshape(1, 3, DIM_STATES[0], DIM_STATES[1])#my_utils.copy_depth_to_3_channel(state_path).reshape(1, 3, DIM_STATES[0], DIM_STATES[1])
        next_state = utils.load_depth_mask_heightmap_data(next_state_path).reshape(1, 3, DIM_STATES[0], DIM_STATES[1])
        # # normlize
        # state = (state - np.min(state)) / (np.max(state) - np.min(state))
        # next_state = (next_state - np.min(next_state)) / (np.max(next_state) - np.min(next_state))
        # numpy to tensor
        state = torch.cuda.FloatTensor(state)
        next_state = torch.cuda.FloatTensor(next_state)

        target_multiChannel_q_map = self.eval_net.forward(state)#dim:[1,CHANNEL,224,224]
        x,y,c = utils.translate_actionID_to_XY_and_channel(action)
        old_val = target_multiChannel_q_map[0][c][x][y]
        # old_val = target[0][action]
        target_val_multiChannel_q_map = self.target_net.forward(next_state)#dim:[1,CHANNEL,224,224]

        if done == 1:
            target_q = reward # target[0][action] = reward
        else:
            target_q = reward + self.discount_factor * torch.max(target_val_multiChannel_q_map) # target[0][action] = reward + self.discount_factor * torch.max(target_val)

        error = abs(old_val - target_q)
        # self.memory.add(error.cpu().detach().numpy(), trans)
        self.memory.add(float(error), trans)

        if reward == 1:
            self.success_sample_action.append(action)
            self.success_sample_state.append(state_path)
            # print('Success sample number: ', len(self.success_sample_action))

    def choose_action(self, state_path, EPSILON):
        # notation that the function return the action's index nor the real action
        state = utils.load_depth_mask_heightmap_data(state_path).reshape(1, 3, DIM_STATES[0], DIM_STATES[1])
        # numpy to tensor
        state = torch.cuda.FloatTensor(state)

        # state = torch.cuda.FloatTensor(my_utils.copy_depth_to_3_channel(state_path).reshape(1,3,DIM_STATES[0],DIM_STATES[1]))#torch.unsqueeze(torch.FloatTensor(state) ,0)
        prob = np.min((EPSILON,1))
        p_select = np.array([prob, 1 - prob])
        # p_select = np.array([0, 1])
        selected_ac_type = np.random.choice([0, 1], p=p_select.ravel())

        if selected_ac_type == 0:#np.random.randn() <= Train_Configs.EPSILON:
            target_multiChannel_q_map = self.eval_net.forward(state)  # dim:[1,CHANNEL,224,224]
            target_multiChannel_q_map = target_multiChannel_q_map[0]  # dim:[CHANNEL,224,224]
            action = utils.find_maxQ_in_qmap(target_multiChannel_q_map.cpu().detach().numpy())
            ac_ty = '0'
            ## check if current action is same as the last one
            if action == self.LAST_ACTION_ID:
                self.SAME_ACTION_COUNT += 1
                if self.SAME_ACTION_COUNT == 4:
                    if np.random.randn() <= 0.5:
                        action = utils.select_randpID_from_mask(state_path)
                        ac_ty = '1'
                    else:
                        action = np.random.randint(0, DIM_ACTIONS)
                        ac_ty = '2'
                    self.SAME_ACTION_COUNT = 0
            else:
                self.LAST_ACTION_ID = action

        else:
            if np.random.randn() <= 0.5:
                action = utils.select_randpID_from_mask(state_path)
                ac_ty = '1'
            else:
                action = np.random.randint(0,DIM_ACTIONS)
                ac_ty = '2'
        #### change the action (id) to robot execution action type

        return ac_ty,action # the id of action

    def choose_action_for_eval(self, state_path):
        # notation that the function return the action's index nor the real action
        state = utils.load_depth_mask_heightmap_data(state_path).reshape(1, 3, DIM_STATES[0], DIM_STATES[1])
        # numpy to tensor
        state = torch.cuda.FloatTensor(state)
        target_multiChannel_q_map = self.eval_net.forward(state)  # dim:[1,CHANNEL,224,224]
        target_multiChannel_q_map = target_multiChannel_q_map[0]  # dim:[CHANNEL,224,224]
        action = utils.find_maxQ_in_qmap(target_multiChannel_q_map.cpu().detach().numpy())

        return action  # the id of action
    
    def plot(self, ax, x):
        ax.cla()
        ax.set_xlabel("episode")
        ax.set_ylabel("total reward")
        ax.plot(x, 'b-')
        plt.pause(0.000000000000001)

    def load_batch_data(self,batch_list):
        # print(batch_list)
        batch_state = []
        batch_action = []
        batch_reward = []
        batch_next_state = []
        batch_mask = []
        for item in batch_list:
            data = item.split('#')#state+'#'+str(action)+'#'+str(reward)+'#'+next_state
            batch_state.append(utils.load_depth_mask_heightmap_data(data[0]).reshape((3,DIM_STATES[0],DIM_STATES[1])))
            batch_action.append([int(data[1])])
            batch_reward.append([float(data[2])])
            batch_next_state.append(utils.load_depth_mask_heightmap_data(data[3]).reshape((3,DIM_STATES[0],DIM_STATES[1])))
            batch_mask.append(utils.load_depth_mask_weights(data[0]).reshape((1,DIM_STATES[0],DIM_STATES[1])))

        # # normlize
        # batch_state = (batch_state - np.min(batch_state)) / (np.max(batch_state) - np.min(batch_state))
        # batch_next_state = (batch_next_state - np.min(batch_next_state)) / (np.max(batch_next_state) - np.min(batch_next_state))

        return torch.cuda.FloatTensor(batch_state),torch.cuda.LongTensor(batch_action),torch.cuda.FloatTensor(batch_reward),torch.cuda.FloatTensor(batch_next_state), batch_mask

    def learn(self):
        # learn 100 times then the target network update
        if self.learn_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_counter+=1

        mini_batch, idxs, is_weights = self.memory.sample(BATCH_SIZE)#
        batch_state, batch_action, batch_reward, batch_next_state, batch_mask = self.load_batch_data(mini_batch)

        eval_multiChannel_q_map = self.eval_net(batch_state)  # dim:[BATCH_SIZE,CHANNEL,224,224]
        x_y_c_list = utils.translate_actionID_to_XY_and_channel_batch(batch_action)
        batch_q = []
        for i in range(len(x_y_c_list)):
            xyc = x_y_c_list[i]
            batch_q.append([eval_multiChannel_q_map[i][xyc[2]][xyc[0]][xyc[1]]])
        q_eval = torch.cuda.FloatTensor(batch_q)#self.eval_net(batch_state).gather(1, batch_action)#action: a value in range [0,DIM_ACTIONS-1]
        q_eval = Variable(q_eval.cuda(), requires_grad=True)
        target_multiChannel_q_map = self.target_net(batch_next_state).cpu().detach().numpy()#q_next,
        batch_q_next = []
        for b_item in target_multiChannel_q_map:
            batch_q_next.append([np.max(b_item)])
        q_next = torch.cuda.FloatTensor(batch_q_next)
        q_target = batch_reward + GAMMA*q_next
        q_target = Variable(q_target.cuda(), requires_grad=True)
        weight_tensor = torch.cuda.FloatTensor(is_weights)#
        weight_tensor = weight_tensor.reshape((BATCH_SIZE,1))
        weight_tensor = Variable(weight_tensor.cuda(), requires_grad=False)

        # loss = (weight_tensor * self.loss(q_eval, q_target)).mean()##(torch.FloatTensor(is_weights) * F.mse_loss(pred, target)).mean()
        loss = (weight_tensor * self.loss(q_eval,q_target))
        loss = loss.sum()
        # # cal the mask loss
        # batch_average_map = torch.mean(eval_multiChannel_q_map, 1, True) # dim:[BATCH_SIZE,CHANNEL,224,224]
        # batch_mask_tensor = Variable(torch.cuda.FloatTensor(batch_mask), requires_grad=False)
        # weighted_batch_q_map = batch_average_map.mul(batch_mask_tensor) #dim:[BATCH_SIZE,1,224,224]
        # mask_loss = (0.5)**(torch.mean(weighted_batch_q_map))
        # # b = a.pow(2)  # 也可以a**2
        # loss = loss + 0.5*mask_loss

        #------- use auxiliary loss ----------#
        if True:
            if len(self.success_sample_action) > 20:
                rand_inds = random.sample(list(range(len(self.success_sample_action))),16)
                # load state and action data
                batch_state = []
                batch_action = []
                for ind in range(len(rand_inds)):
                    batch_state.append(utils.load_depth_mask_heightmap_data(self.success_sample_state[rand_inds[ind]]).reshape((3,DIM_STATES[0],DIM_STATES[1])))
                    batch_action.append(self.success_sample_action[rand_inds[ind]])
                
                batch_state = torch.cuda.FloatTensor(batch_state)
                eval_multiChannel_q_map = self.eval_net(batch_state)  # dim:[BATCH_SIZE,CHANNEL,224,224]
                output_map_tensor = Variable(eval_multiChannel_q_map.cuda(), requires_grad=True)
                # process action data
                x_y_c_list = utils.translate_actionID_to_XY_and_channel_batch(batch_action)
                batch_action_label = torch.zeros([16,8,224,224],dtype=torch.float)
                for i in range(16):
                    x,y,c = x_y_c_list[i]
                    for j in range(8):
                        if j == c:
                            batch_action_label[i,j,x,y] = 0.79
                        else:
                            batch_action_label[i,j,x,y] = 0.03
                batch_action_label = Variable(batch_action_label.cuda(), requires_grad=True)
                mse_loss_fun = torch.nn.MSELoss()
                mse_loss = mse_loss_fun(output_map_tensor, batch_action_label)
                mse_loss*=50
                # add to total loss
                loss = loss + mse_loss
        #-------------------------------------#

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss),float(q_eval.mean()),float(mse_loss)

if __name__ == '__main__':
    pass
    # main()
    # train_from_checkpoint('step_115000_params.pkl',115000)
    # train_from_checkpoint('step_75000_params.pkl',75000)
