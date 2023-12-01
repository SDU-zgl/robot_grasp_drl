import numpy as np
import vrep
import random
import cv2
import time
import os

class Robot(object):
    def __init__(self,workspace_limits):

        self.workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])

        self.color_space = np.asarray([[78.0, 121.0, 167.0], # blue
                                        [89.0, 161.0, 79.0], # green
                                        [156, 117, 95], # brown
                                        [242, 142, 43], # orange
                                        [237.0, 201.0, 72.0], # yellow
                                        [186, 176, 172], # gray
                                        [255.0, 87.0, 89.0], # red
                                        [176, 122, 161], # purple
                                        [118, 183, 178], # cyan
                                        [255, 157, 167]])/255.0 #pink

        # Randomly choose objects to add to scene
        # self.obj_mesh_ind = np.random.randint(0, len(self.mesh_list), size=self.num_obj)
        self.obj_mesh_color = self.color_space[np.asarray(range(8)) % 10, :]

        # Define the obj path for importing object meshes
        self.obj_mesh_dir = 'objects/'
        self.emptyBuff = bytearray()
        self.running = True
        self.task_count = 0

        self.object_name_list = ['object1','object2','object3','object4','object5','object6']

        # Make sure to have the server side running in V-REP:
        # Connect to simulator
        vrep.simxFinish(-1) # Just in case, close all opened connections
        self.sim_client = vrep.simxStart('127.0.0.1', 19997, True, True, -500000, 5) # Connect to V-REP on port 19997
        if self.sim_client == -1:
            print('Failed to connect to simulation (V-REP remote API server). Exiting.')
            exit()
        else:
            print('Connected to simulation, initializing...')
            # Start the simulation:
            vrep.simxStartSimulation(self.sim_client,vrep.simx_opmode_oneshot_wait)
            time.sleep(4)

        res,self.robotHandle=vrep.simxGetObjectHandle(self.sim_client,'UR5',vrep.simx_opmode_oneshot_wait)
        # Find the camera handle
        sim_ret, self.rgb_cam_handle = vrep.simxGetObjectHandle(self.sim_client, 'kinect_rgb', vrep.simx_opmode_blocking)
        sim_ret, self.dep_cam_handle = vrep.simxGetObjectHandle(self.sim_client, 'kinect_depth', vrep.simx_opmode_blocking)

    def wait_for_execution(self):
        # Wait until the end of the movement:
        self.running = True
        while self.running:
            res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(self.sim_client,'remoteApiCommandServer',vrep.sim_scripttype_childscript,'isRunning',[self.robotHandle],[],[],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
            self.running=retInts[0]==1

    def move_to_position(self,position):
        '''
        Description: move the robot to given position
        Params: position, [x,y,z,q1,q2,q3,q4], (x,y,z) is the coordinate in world coordinate system, (q1,q2,q3,q4) is the quaternion that defines the target orientation
        '''
        res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(self.sim_client,'remoteApiCommandServer',vrep.sim_scripttype_childscript,'runActionfromClient',[self.robotHandle],position,[],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
        self.wait_for_execution()

    def go_to_reset_position(self):
        self.move_to_position(np.array([-0.2, -0.1, 0.3,  0.000000e+00, 1.000000e+00,  0.000000e+00,  6.123234e-17]))

    def get_kinect_data(self):
        ######## get color(rgb) data, 640*480 #########
        sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.rgb_cam_handle, 0, vrep.simx_opmode_blocking)
        color_img = np.asarray(raw_image)
        color_img.shape = (resolution[1], resolution[0], 3)
        color_img = color_img.astype(np.float64)/255
        color_img[color_img < 0] += 1
        color_img *= 255
        color_img = np.fliplr(color_img)
        color_img = color_img.astype(np.uint8)
        color_img = cv2.cvtColor(color_img,cv2.COLOR_RGB2BGR)

        ######## get depth data, 640*480 #########
        sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.sim_client, self.dep_cam_handle, vrep.simx_opmode_blocking)
        depth_img = np.asarray(depth_buffer)
        depth_img.shape = (resolution[1], resolution[0])
        depth_img = np.fliplr(depth_img)
        zNear = 0.01
        zFar = 3.5
        depth_img = depth_img * (zFar - zNear) + zNear #mm

        return color_img, depth_img

    def get_current_state(self,flag):
        color_img, depth_img = self.get_kinect_data()
        color_img = color_img[36:436, 120:520, :] #640*480
        depth_img = depth_img[36:436, 120:520]
        color_img = cv2.resize(color_img,(224,224))
        depth_img = cv2.resize(depth_img,(224,224))
        file_list = os.listdir('sampled_state/'+flag+'/')
        file_num = int(len(file_list)/3.0)
        cv2.imwrite('sampled_state/'+flag+'/rgb_'+str(file_num+1)+'.png', cv2.rotate(color_img, cv2.ROTATE_180))
        np.save('sampled_state/'+flag+'/depth_'+str(file_num+1)+'.npy', cv2.rotate(depth_img, cv2.ROTATE_180))
        return 'sampled_state/'+flag+'/depth_'+str(file_num+1)+'.npy'

    # open rg2
    def openRG2(self):
        rgName = 'RG2'
        clientID = self.sim_client
        res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(clientID, rgName, vrep.sim_scripttype_childscript,'rg2Open',[],[],[],b'',vrep.simx_opmode_blocking)
        time.sleep(1.5)

    # close rg2
    def closeRG2(self):
        rgName = 'RG2'
        clientID = self.sim_client
        res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(clientID, rgName, vrep.sim_scripttype_childscript,'rg2Close',[],[],[],b'',vrep.simx_opmode_blocking)
        time.sleep(1.5)

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
            object_position = [drop_x, drop_y, 0.01]
            # nishizhen positive; shunshizhen negtive
            rand_angle = np.random.randint(-60, 60)
            sim_ret, object_orientation = vrep.simxGetObjectOrientation(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            object_orientation[2] = rand_angle/180.0*np.pi#[-np.pi/2, 0, rand_angle/180.0*np.pi]
            vrep.simxSetObjectPosition(self.sim_client, object_handle, -1, object_position, vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, object_handle, -1, object_orientation, vrep.simx_opmode_blocking)
            time.sleep(0.5)

        self.task_count += 1
        self.go_to_reset_position()

    def generate_random_position(self):
        random_pos_list_1 = [[-0.6493, -0.112],[-0.5, -0.112],[-0.35066, -0.112],[-0.6493, 0.112],[-0.5, 0.112],[-0.35066, 0.112]]
        random_pos_list_2 = [[-0.612, -0.1493],[-0.612, 0],[-0.612, 0.1493],[-0.388, -0.1493],[-0.388, 0],[-0.388, 0.1493]]
        if self.task_count % 2 == 0:
            pos_list = random_pos_list_1
        else:
            pos_list = random_pos_list_2

        pos1 = [pos_list[0][0]+random.uniform(-0.03,0.03), pos_list[0][1]+random.uniform(-0.03,0.03)]
        pos2 = [pos_list[1][0]+random.uniform(-0.03,0.03), pos_list[1][1]+random.uniform(-0.03,0.03)]
        pos3 = [pos_list[2][0]+random.uniform(-0.03,0.04), pos_list[2][1]+random.uniform(-0.03,0.04)]
        pos4 = [pos_list[3][0]+random.uniform(-0.03,0.03), pos_list[3][1]+random.uniform(-0.03,0.03)]
        pos5 = [pos_list[4][0]+random.uniform(-0.03,0.04), pos_list[4][1]+random.uniform(-0.03,0.04)]
        pos6 = [pos_list[5][0]+random.uniform(-0.03,0.03), pos_list[5][1]+random.uniform(-0.04,0.03)]

        return [pos1, pos2, pos3, pos4, pos5, pos6]

    def check_grasp_success(self):
        flag = False
        grasped_object_handle = 0
        for i in range(len(self.object_name_list)):
            obj_name = self.object_name_list[i]
            sim_ret, object_handle = vrep.simxGetObjectHandle(self.sim_client, obj_name, vrep.simx_opmode_blocking)
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            if object_position[2] > 0.15:
                flag = True
                grasped_object_handle = object_handle
        if flag:
            drop_x = -0.5+random.uniform(-0.12,0.12)
            drop_y = -0.55+random.uniform(-0.08,0.08)
            object_position = [drop_x, drop_y, 0.15]
            object_orientation = [-np.pi/2, 0, 0]
            vrep.simxSetObjectPosition(self.sim_client, grasped_object_handle, -1, object_position, vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, grasped_object_handle, -1, object_orientation, vrep.simx_opmode_blocking)
            time.sleep(1)

        return flag

    def check_scene_empty(self):
        workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])
        is_empty = True
        for i in range(len(self.object_name_list)):
            obj_name = self.object_name_list[i]
            sim_ret, object_handle = vrep.simxGetObjectHandle(self.sim_client, obj_name, vrep.simx_opmode_blocking)
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            if object_position[0] > workspace_limits[0][0] and object_position[0] < workspace_limits[0][1] and object_position[1] > workspace_limits[1][0] and object_position[1] < workspace_limits[1][1]:
                is_empty = False

        return is_empty
    # def reposition_objects(self, workspace_limits):
    #     # Move gripper out of the way
    #     self.move_to([-0.1, 0, 0.3], None)

    #     for object_handle in self.object_handles:

    #         # Drop object at random x,y location and random orientation in robot workspace
    #         drop_x = (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * np.random.random_sample() + workspace_limits[0][0] + 0.1
    #         drop_y = (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * np.random.random_sample() + workspace_limits[1][0] + 0.1
    #         object_position = [drop_x, drop_y, 0.15]
    #         object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
    #         vrep.simxSetObjectPosition(self.sim_client, object_handle, -1, object_position, vrep.simx_opmode_blocking)
    #         vrep.simxSetObjectOrientation(self.sim_client, object_handle, -1, object_orientation, vrep.simx_opmode_blocking)
    #         time.sleep(2)

    # def get_static_object_position_and_angle(self):
    #     obj_positions = []
    #     obj_orientations = []
    #     for name in self.object_name_list:
    #         sim_ret, tmp_handle = vrep.simxGetObjectHandle(self.sim_client, name, vrep.simx_opmode_blocking)
    #         sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, tmp_handle, -1, vrep.simx_opmode_blocking)
    #         sim_ret, object_orientation = vrep.simxGetObjectOrientation(self.sim_client, tmp_handle, -1, vrep.simx_opmode_blocking)
    #         obj_positions.append(object_position)
    #         obj_orientations.append(object_orientation)

    #     return obj_positions, obj_orientations