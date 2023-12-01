import math
import numpy as np
import cv2
from scipy.spatial.transform import Rotation


DIM_STATES = (224, 224)

def rotEuler2Others(euler_angle):
    rot_euler = Rotation.from_euler('ZYX', euler_angle)
    # q = rot_gate.as_quat()
    quat = rot_euler.as_quat()
    rotvec = rot_euler.as_rotvec()
    rotmatrix = rot_euler.as_matrix()
    # print('quat: ',quat)
    return quat

def rotMatrix2Others(rotation_matrix):
    r = Rotation.from_matrix(rotation_matrix)
    quat = r.as_quat()
    rotvec = r.as_rotvec()
    euler = r.as_euler('zyx', degrees=True)
    return quat, rotvec, euler

def rotMatrix2rotVector(R):
    assert (isRotm(R))
    res, _ = cv2.Rodrigues(R)
    return res

def rotVector2rotMatrix(Rvec):
    res, _ = cv2.Rodrigues(Rvec)
    return res

# Get rotation matrix from euler angles
def euler2rotm(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

# Checks if a matrix is a valid rotation matrix.
def isRotm(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
def rotm2euler(R):
    assert (isRotm(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def angle2rotm(angle, axis, point=None):
    # Copyright (c) 2006-2018, Christoph Gohlke

    sina = math.sin(angle)
    cosa = math.cos(angle)
    axis = axis / np.linalg.norm(axis)

    # Rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(axis, axis) * (1.0 - cosa)
    axis *= sina
    R += np.array([[0.0, -axis[2], axis[1]],
                   [axis[2], 0.0, -axis[0]],
                   [-axis[1], axis[0], 0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # Rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M

def rotm2angle(R):
    # From: euclideanspace.com

    epsilon = 0.01  # Margin to allow for rounding errors
    epsilon2 = 0.1  # Margin to distinguish between 0 and 180 degrees

    assert (isRotm(R))

    if ((abs(R[0][1] - R[1][0]) < epsilon) and (abs(R[0][2] - R[2][0]) < epsilon) and (
            abs(R[1][2] - R[2][1]) < epsilon)):
        # Singularity found
        # First check for identity matrix which must have +1 for all terms in leading diagonaland zero in other terms
        if ((abs(R[0][1] + R[1][0]) < epsilon2) and (abs(R[0][2] + R[2][0]) < epsilon2) and (
                abs(R[1][2] + R[2][1]) < epsilon2) and (abs(R[0][0] + R[1][1] + R[2][2] - 3) < epsilon2)):
            # this singularity is identity matrix so angle = 0
            return [0, 1, 0, 0]  # zero angle, arbitrary axis

        # Otherwise this singularity is angle = 180
        angle = np.pi
        xx = (R[0][0] + 1) / 2
        yy = (R[1][1] + 1) / 2
        zz = (R[2][2] + 1) / 2
        xy = (R[0][1] + R[1][0]) / 4
        xz = (R[0][2] + R[2][0]) / 4
        yz = (R[1][2] + R[2][1]) / 4
        if ((xx > yy) and (xx > zz)):  # R[0][0] is the largest diagonal term
            if (xx < epsilon):
                x = 0
                y = 0.7071
                z = 0.7071
            else:
                x = np.sqrt(xx)
                y = xy / x
                z = xz / x
        elif (yy > zz):  # R[1][1] is the largest diagonal term
            if (yy < epsilon):
                x = 0.7071
                y = 0
                z = 0.7071
            else:
                y = np.sqrt(yy)
                x = xy / y
                z = yz / y
        else:  # R[2][2] is the largest diagonal term so base result on this
            if (zz < epsilon):
                x = 0.7071
                y = 0.7071
                z = 0
            else:
                z = np.sqrt(zz)
                x = xz / z
                y = yz / z
        return [angle, x, y, z]  # Return 180 deg rotation

    # As we have reached here there are no singularities so we can handle normally
    s = np.sqrt(
        (R[2][1] - R[1][2]) * (R[2][1] - R[1][2]) + (R[0][2] - R[2][0]) * (R[0][2] - R[2][0]) + (R[1][0] - R[0][1]) * (
                    R[1][0] - R[0][1]))  # used to normalise
    if (abs(s) < 0.001):
        s = 1

        # Prevent divide by zero, should not happen if matrix is orthogonal and should be
    # Caught by singularity test above, but I've left it in just in case
    angle = np.arccos((R[0][0] + R[1][1] + R[2][2] - 1) / 2)
    x = (R[2][1] - R[1][2]) / s
    y = (R[0][2] - R[2][0]) / s
    z = (R[1][0] - R[0][1]) / s
    return [angle, x, y, z]

def merge_center(center1, center2):
    # the centers detected using depth are correct 
    return center1

def merge_angle(angle1, angle2):
    # the angles detected using depth are correct 
    return angle1

def object_grasp_detection(rgb, depth):
    # crop the rgb image
    rgb = rgb[36:436, 120:520, :] #640*480
    rgb = cv2.rotate(rgb, cv2.ROTATE_180)
    depth = depth[36:436, 120:520]
    # find the object edge using background difference
    max_depth_value = np.max(depth)
    raw_mask = max_depth_value - 0.01 - depth
    raw_mask = np.clip(raw_mask, 0, 255)
    raw_mask = np.clip(raw_mask*100000, 0, 255)
    # rotate the image to align with the real world
    rotate_mask = cv2.rotate(raw_mask, cv2.ROTATE_180)
    cv2.imwrite('binary_mask.png',rotate_mask)
    binary_mask = cv2.imread('binary_mask.png', 0)
    # find grasp solution on binary mask of depth
    ret, binary = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    object_centers = []
    object_angles = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # exclude the noise point
        if w>=15 and h>= 15:
            # find the rotation angle
            rect = cv2.minAreaRect(cnt)
            object_centers.append([int(rect[0][0]),int(rect[0][1])])
            object_angles.append(rect[2])
            ### debug ###
            points_rect = cv2.boxPoints(rect)
            box = np.int0(points_rect)
            rgb = cv2.drawContours(rgb, [box], 0, (0, 0, 255), 2)
            #############
    cv2.imwrite('rgb_vis.png',rgb)

    # find grasp solution on binary mask of rgb
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(rgb, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    object_centers_ = []
    object_angles_ = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # exclude the noise point
        if w>=10 and h>= 10:
            object_centers_.append([int(x+w/2.0),int(+h/2.0)])
            # find the rotation angle
            rect = cv2.minAreaRect(cnt)
            object_angles_.append(rect[2])

    return merge_center(object_centers,object_centers_), merge_angle(object_angles,object_angles_)

def image_coordinate_to_robot_coordinate(image_pos):
    u, v = image_pos[0], image_pos[1]
    u = u*400/224.0
    v = v*400/224.0
    # 411pix <-> 0.51m
    resolution_image_to_world = 0.51 / 411.0
    pos_x = -0.5 - (v-200)*resolution_image_to_world
    pos_y = (200-u)*resolution_image_to_world
    
    return pos_x, pos_y


#***********************************************************************************************************************************
def depth_npy2mask():
    pass

def actionID_to_depthImgXY(action_id):
    dim_for_one_direction = 224 * 224
    angle_id = int(action_id / dim_for_one_direction)
    action_oz = angleId_to_angle(angle_id)
    # trans to depth imageXY
    new_id = action_id % dim_for_one_direction
    h = int(new_id / 224) + 1
    w = new_id % 224 + 1
    # (col,row)
    depth_x, depth_y = w, h
    return (depth_x,depth_y)#(col,row)
def depthImgXY_to_actionID_withRandAngle(pos):
    dim_for_one_direction = 224 * 224
    h = pos[1]
    w = pos[0]
    id_temp = (w-1) + (h-1)*224
    rand_angle = np.random.randint(0,8)
    id = id_temp + rand_angle * dim_for_one_direction
    # print('pack action id: (id,w,h,angle): ',id,w,h,rand_angle)
    return int(id)

def predict_actionID_to_execution_action(action_id,current_depth_path):
    # actionID [0~224*224*10-1]
    # the id increase from top to bottom and from left to right, and return the position(x,y) in format (w,h), [1~224]
    dim_for_one_direction = 224*224
    angle_id = int(action_id/dim_for_one_direction)
    action_oz = angleId_to_angle(angle_id)
    # trans to depth imageXY
    new_id = action_id%dim_for_one_direction
    h = int(new_id/224) + 1
    w = new_id%224 + 1
    #(col,row)
    depth_x,depth_y = w,h
    # print('unpack action id (id,w,h,angle): ',action_id,w,h,angle_id)
    # action_x, action_y = imageXY_to_robotXY((depth_x, depth_y))
    action_x, action_y = image_coordinate_to_robot_coordinate((depth_x, depth_y))
    action_z = cal_corresponding_z_fromXY((depth_x, depth_y), current_depth_path)

    # return_x = work_space_x[1] - (p[0] - 1) * (abs(work_space_x[1] - work_space_x[0]) / 226.0)
    # return_y = work_space_y[0] + (p[1] - 1) * ((work_space_y[1] - work_space_y[0]) / 226.0)

    # return (return_x, return_y)  # imageXY format: (width,height)
    action = (action_x, action_y, action_z, action_oz[0], action_oz[1], action_oz[2], action_oz[3])
    return action

def cal_corresponding_z_fromXY(XY,depth_path):#(col,row)
    # print('XY:',XY)
    data=np.load(depth_path)
    depth_value = data[int(XY[0])-1][int(XY[1])-1]

    Z = (0.90788731274057 - depth_value) / 1.07428894499597
    t = Z/2.0 - 0.0065
    if t >= 0.01:
        re = t
    else:
        re = 0.01
    return re

def select_randP_from_mask(npy_path):
    # no-object value:0.908,0.907, 0.7307463443279267
    # object value:0.86
    xy_contrain_object = []  # record the pixel of object
    max_depth_value = 0.7307463443279267
    data = np.load(npy_path)
    h, w = data.shape
    raw_mask = max_depth_value - 0.01 - data
    raw_mask = np.clip(raw_mask, 0, 255)
    raw_mask = np.clip(raw_mask*100000, 0, 255)

    for i in range(w):
        for j in range(h):
            if raw_mask[i][j] == 255:
                xy_contrain_object.append((j, i)) #w-h
    if len(xy_contrain_object) != 0:
        rand_id = np.random.randint(0, len(xy_contrain_object) - 1)
        return xy_contrain_object[rand_id]# return imageXY format: (width,height)
    else:
        print('there is not valid point detected !')
        # return (random.randint(10,217),random.randint(10,217))
        return -1

def select_randpID_from_mask(npy_path):
    p = select_randP_from_mask(npy_path)
    if p == -1:
        return int(224*224/2)#-1#np.random.randint(0,_configs.DIM_ACTIONS)
    else:
        id = depthImgXY_to_actionID_withRandAngle(p)
        return id

def imageXY_to_robotXY(p):
    # work_space = _configs.WORKSPACE_LIMITS
    # work_space_x = work_space[0]
    # work_space_y = work_space[1]
    #
    # return_x = work_space_x[0] + (p[0] - 1) * -(abs(work_space_x[1] - work_space_x[0]) / 226.0)
    # return_y = work_space_y[0] + (p[1] - 1) * ((work_space_y[1] - work_space_y[0]) / 226.0)
    #
    # return (return_x,return_y)
    # WORKSPACE_LIMITS = np.asarray([[-0.705, -0.195], [-0.255, 0.255], [0.01, 0.3]])
    work_space = WORKSPACE_LIMITS
    work_space_x = work_space[0]
    work_space_y = work_space[1]

    return_x = work_space_x[1] - (p[0] - 1) * (abs(work_space_x[1] - work_space_x[0]) / 223.0)
    return_y = work_space_y[0] + (p[1] - 1) * ((work_space_y[1] - work_space_y[0]) / 223.0)

    return (return_x, return_y) # imageXY format: (width,height)

def robotXY_to_imageXY(p):
    # work_space = _configs.WORKSPACE_LIMITS
    # work_space_x = work_space[0]
    # work_space_y = work_space[1]
    #
    #
    # return_y = (226 * (p[0]-work_space_x[0]) + (work_space_x[1] - work_space_x[0])) / (work_space_x[1] - work_space_x[0])#action_space_x[0] + (p[0] - 1) * ((action_space_x[1] - action_space_x[0]) / 226.0)
    # return_x = (226 * (p[1]-work_space_y[0]) + (work_space_y[1] - work_space_y[0])) / (work_space_y[1] - work_space_y[0])
    #
    # return (return_x,return_y)
    work_space = _configs.WORKSPACE_LIMITS
    work_space_x = work_space[0]
    work_space_y = work_space[1]

    return_w = ((abs(p[0]) - abs(work_space_x[1])) / abs(work_space_x[1] - work_space_x[0])) * 224
    return_h = (p[1] - work_space_y[0]) / abs(work_space_y[1] - work_space_y[0]) * 224
    if return_w == 0:
        return_w = 1
    if return_h == 0:
        return_h = 1

    return (return_w, return_h)  # imageXY format: (width,height)

def output_action_2_execution_action(x, y):
    real_x = _configs.ACTION_SPACE_GRASP[0][1] - 0.5 * (1 - x) * (_configs.ACTION_SPACE_GRASP[0][1] - _configs.ACTION_SPACE_GRASP[0][0])
    real_y = _configs.ACTION_SPACE_GRASP[1][1] - 0.5 * (1 - y) * (_configs.ACTION_SPACE_GRASP[1][1] - _configs.ACTION_SPACE_GRASP[1][0])

    return real_x, real_y


def execution_action_2_output_action(x, y):
    out_x = 1 - 2.0 * (_configs.ACTION_SPACE_GRASP[0][1] - x) / (_configs.ACTION_SPACE_GRASP[0][1] - _configs.ACTION_SPACE_GRASP[0][0])
    out_y = 1 - 2.0 * (_configs.ACTION_SPACE_GRASP[1][1] - y) / (_configs.ACTION_SPACE_GRASP[1][1] - _configs.ACTION_SPACE_GRASP[1][0])

    return out_x, out_y


def angleId_to_angle(id):# the range (0~180), not (0~360)
    temp = (180/8) * id
    tmp_quat = rotEuler2Others([(90-abs(temp)), 0, -np.pi])

    return tmp_quat#temp * np.pi / 180.0

def angle_to_angleId(ang):# the range (0~180), not (0~360)
    angle_index = np.array(list(range(16))) / 16.0 * np.pi
    # print('id:',np.argmin(abs(angle_index-ang)))
    return np.argmin(abs(angle_index-ang))

#----------------- functions for multi-channel FCN DQN -------------------
def translateXY_and_channel_to_actionID(x,y,c):
    '''
    given a state, the eval_net produce the multi-channel q-value map (c,224,224), then find the max q-value, get its position(x,y) and channel c
    :param x: h, the row
    :param y: w, the col
    :param c: channel
    :return: id, rule: from left to right, from top to bottom, the id is (0~224*224*c)
    '''
    im_size = DIM_STATES[0]
    margin = im_size*im_size
    return x * im_size + y + margin*c

def translate_actionID_to_XY_and_channel(action_id):
    '''
    action_id => (x,y,c)
    :param x: h, the row
    :param y: w, the col
    :param c: channel
    :return: id, rule: from left to right, from top to bottom, the id is (0~224*224*c)
    '''
    im_size = DIM_STATES[0]
    margin = im_size*im_size
    c = int(action_id/margin)
    action_id0 = action_id%margin
    x = int(action_id0/im_size)
    y = action_id0 % im_size
    return x,y,c

def translate_actionID_to_XY_and_channel_batch(batch_action_id):
    result_list = []
    for action_id in batch_action_id:
        [x,y,c] = translate_actionID_to_XY_and_channel(action_id)
        result_list.append([x,y,c])
    return result_list

def find_maxQ_in_qmap(q_map):
    '''
    :param q_map: [CHANNEL,224,224]
    :return: action_id with the max q value
    '''

    a = q_map
    m, n, l = a.shape
    index = int(a.argmax())
    x = int(index / (n * l))
    index = index % (n * l)
    y = int(index / l)
    index = index % l
    z = index
    return translateXY_and_channel_to_actionID(y,z,x)#(x, y, z)
def copy_depth_to_3_channel(state_path):
    one_dim_arr = np.load(state_path)
    return np.stack((one_dim_arr, one_dim_arr, one_dim_arr))

def get_heightmap_from_depth(depth_arr,mask_arr):
    height_map_arr = mask_arr
    bottom_mean = 0.9078791
    h_inds,w_inds = np.where(mask_arr==255)
    for i in range(len(h_inds)):
        height_map_arr[h_inds[i]][w_inds[i]] = round(( bottom_mean - mask_arr[h_inds[i]][w_inds[i]] ) *1000) # mm level

    height_map_arr = (height_map_arr - _configs.MIN_HEIGHTMAP_ARR) / (_configs.MAX_HEIGHTMAP_ARR - _configs.MIN_HEIGHTMAP_ARR)
    return np.clip(height_map_arr, 0, 100)

def load_depth_mask_heightmap_data(state_path):
    # mask
    max_depth_value = 0.7307463443279267
    origin_depth_arr = np.load(state_path)
    raw_mask = max_depth_value - 0.01 - origin_depth_arr
    raw_mask = np.clip(raw_mask, 0, 255)
    raw_mask = np.clip(raw_mask*100000, 0, 255)
    cv2.imwrite(state_path.replace('npy','png'),raw_mask)
    mask_arr = raw_mask / 255.0
    
    # rgb, depth_1.npy, rgb_1.png
    rgb = cv2.imread(state_path.replace('depth','rgb').replace('npy','png'),0)
    rgb_arr = rgb / 255.0

    return np.stack((mask_arr, origin_depth_arr, rgb_arr))
    # return np.stack((mask_arr, mask_arr, mask_arr))

def load_depth_mask_heightmap_data_onlyMask(state_path):
    
    max_depth_value = 0.7307463443279267
    origin_depth_arr = np.load(state_path)
    raw_mask = max_depth_value - 0.01 - origin_depth_arr
    raw_mask = np.clip(raw_mask, 0, 255)
    raw_mask = np.clip(raw_mask*100000, 0, 255)
    cv2.imwrite(state_path.replace('npy','png'),raw_mask)
    mask_arr = raw_mask / 255.0

    return np.stack((mask_arr, mask_arr, mask_arr))

def load_depth_mask_weights(state_path):
    # no-object value:0.908,0.907, new: 0.7307463443279267
    # object value:0.86
    
    origin_depth_arr = np.load(state_path)

    # mask
    mask_arr = (origin_depth_arr - 0.726) * -1000000
    mask_arr = np.clip(mask_arr, 0, 255)
    mask_arr = mask_arr / 255.0

    return mask_arr#np.stack((depth_arr, mask_arr, heightmap_arr))

def select_random_obj(num):
    # all_obj_type = ['0.obj','1.obj','2.obj','3.obj','4.obj','6.obj','7.obj','8.obj','ball_017.obj','ball_018.obj','ball_019.obj','ball_020.obj','cube.obj','cube_027.obj','cube_028.obj','cuboid_025.obj','cuboid_026.obj','cuboid_029.obj','cuboid_032.obj','cuboid_033.obj','cuboid_034.obj','cuboid_036.obj','cuboid_038.obj','cylinder_001.obj','cylinder_002.obj','cylinder_003.obj','cylinder_004.obj','cylinder_005.obj','cylinder_006.obj','cylinder_007.obj','cylinder_008.obj','cylinder_009.obj','cylinder_010.obj','cylinder_011.obj','cylinder_012.obj','cylinder_013.obj','cylinder_014.obj','cylinder_017.obj','cylinder_018.obj','cylinder_019.obj','cylinder_020.obj','dumbbell_021.obj','dumbbell_022.obj','dumbbell_023.obj','dumbbell_024.obj','efiber_012.obj','efiber_013.obj','efiber_014.obj','lego1.obj','lego2.obj','lego3.obj','lego4.obj','lego6.obj','lego7.obj','Lshape_039.obj','Lshape_040.obj','Lshape_041.obj','Lshape_042.obj','Lshape_043.obj','six_prism_001.obj','six_prism_002.obj','six_prism_003.obj','six_prism_009.obj','six_prism_010.obj','six_prism_011.obj']
    all_obj_type = ['0.obj','1.obj','2.obj','3.obj','4.obj','6.obj','7.obj','8.obj','ball','cube','cuboid','cylinder','dumbbell','efiber','lego','Lshape','six_prism']
    ball_obj = ['ball_017.obj','ball_018.obj','ball_019.obj','ball_020.obj']
    cube_obj = ['cube.obj','cube_027.obj','cube_028.obj']
    cuboid_obj = ['cuboid_025.obj','cuboid_026.obj','cuboid_029.obj','cuboid_032.obj','cuboid_033.obj','cuboid_034.obj','cuboid_036.obj']
    cylinder_obj = ['cylinder_001.obj','cylinder_002.obj','cylinder_003.obj','cylinder_004.obj','cylinder_005.obj','cylinder_006.obj','cylinder_007.obj','cylinder_008.obj','cylinder_009.obj','cylinder_010.obj','cylinder_011.obj','cylinder_012.obj','cylinder_013.obj','cylinder_014.obj','cylinder_017.obj','cylinder_018.obj','cylinder_019.obj','cylinder_020.obj']
    dumbbell_obj = ['dumbbell_021.obj','dumbbell_022.obj','dumbbell_023.obj','dumbbell_024.obj']
    efiber_obj = ['efiber_012.obj','efiber_013.obj','efiber_014.obj']
    lego_obj = ['lego1.obj','lego2.obj','lego3.obj','lego4.obj','lego6.obj','lego7.obj']
    Lshape_obj = ['Lshape_039.obj','Lshape_040.obj','Lshape_041.obj','Lshape_042.obj','Lshape_043.obj']
    six_prism_obj = ['six_prism_001.obj','six_prism_002.obj','six_prism_003.obj','six_prism_009.obj','six_prism_010.obj','six_prism_011.obj']

    # 'ball','cube','cuboid','cylinder','dumbbell','efiber','lego','Lshape','six_prism'
    obj_dict = {}
    obj_dict['ball'] = ball_obj
    obj_dict['cube'] = cube_obj
    obj_dict['cuboid'] = cuboid_obj
    obj_dict['cylinder'] = cylinder_obj
    obj_dict['dumbbell'] = dumbbell_obj
    obj_dict['efiber'] = efiber_obj
    obj_dict['lego'] = lego_obj
    obj_dict['Lshape']= Lshape_obj
    obj_dict['six_prism'] = six_prism_obj

    all_keys = obj_dict.keys()
    select_obj = random.sample(all_obj_type,num)
    for ind in range(num):
        if select_obj[ind] in all_keys:
            replace_obj_item = random.sample(obj_dict[select_obj[ind]],1)[0]
            select_obj[ind] = replace_obj_item

    # print(select_obj)
    return select_obj

def select_random_obj_new(num):
    base_path = 'home/ys/...'
    base_type_list = ['basic_fruit/','block/','flashlight/','food_box/','fruit/','glue/','shampoo_bottle/','soda_can/']
    rand_selected_type_list = random.sample(base_type_list,num)
    return_list = []
    for item in rand_selected_type_list:
        all_file_list = os.listdir(base_path+item)
        single_rand_selected_obj = random.sample(all_file_list,1)[0]
        return_list.append(item+single_rand_selected_obj)

    return return_list


######## test #########
# R = np.array([[],
#               [],
#               []])

# R_vec = np.array([[2.22],
#                   [-2.22],
#                   [0]])

# R_vec = np.array([2.22, -2.22, 0])

# R = rotVector2rotMatrix(R_vec)
# print('Rot max:')
# print(R)
# print('Angle:')
# print(rotm2angle(R))
# print('euler:')
# print(rotm2euler(R) / np.pi * 180)
# print('************************')
# quat, rotvec, euler = rotMatrix2Others(R)
# print('new quat: ')
# print(quat)
# print('new rotvec: ')
# print(rotvec)
# print('new euler: ')
# print(euler)

# euler = np.array([-9.0000e+01, +8.1422e-13, -1.8000e+02])
# quat, rotvec, rotmatrix = rotEuler2Others(euler)
# print(quat)



# rgb = cv2.imread('rgb_demo.png')
# depth = np.load('dep_demo.npy')
# center1,angle1,center2,angle2 = object_grasp_detection(rgb,depth)
# print('center1: ')
# print(center1)
# print('angle1: ')
# print(angle1)
# print('center2: ')
# print(center2)
# print('angle2: ')
# print(angle2)