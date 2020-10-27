#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
#import jsk_2020_4_carry_dish
#from scripts import Net
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import os
import rospy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
import tf
import random
import time
import datetime
from sensor_msgs import point_cloud2 
import pickle
from PIL import Image

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        """
        This imitates alexnet. 
        self.conv1 = nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2) #入力チャンネル数は1, 出力チャンネル数は96 
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        #self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc1 = nn.Linear(50176, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)
        self.fc4 = nn.Linear(10 + 4, 14)
        self.fc5 = nn.Linear(14, 1) # output is 1 dim scalar probability
        """

        # dynamics-net (icra2019の紐とか柔軟物を操るやつ) by Mr. Kawaharazuka
        self.conv1 = nn.Conv2d(2, 4, 3, 2, 1)
        self.cbn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, 3, 2, 1)
        self.cbn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, 3, 2, 1)
        self.cbn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, 3, 2, 1)
        self.cbn4 = nn.BatchNorm2d(32)
        #self.conv5 = nn.Conv2d(32, 64, 3, 2, 1)
        #self.cbn5 = nn.BatchNorm2d(64)
        #self.fc1 = nn.Linear(256, 64)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8 + 4, 12)
        self.fc5 = nn.Linear(12, 1) # output is 1 dim scalar probability

    # depth encording without concate grasp point
    def forward(self, x, y):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = self.cbn1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.cbn2(x)
        x = F.relu(self.conv3(x))
        x = self.cbn3(x)
        x = F.relu(self.conv4(x))
        x = self.cbn4(x)
        #x = F.max_pool2d(F.relu(self.conv5(x)), 2)
        #x = self.cbn5(x)
        x = x.view(-1, self.num_flat_features(x))
        #depth_data =depth_data.view(depth_data.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        z = torch.cat((x, y), dim=1)
        z = F.relu(self.fc4(z))
        z = self.fc5(z)
        return z
   
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class TestSystem():
    def __init__(self):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_path = 'model.pth'
        self.model = torch.load(model_path)
        self.criterion = nn.BCEWithLogitsLoss()
        self.test_optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    def load_model(self):
        # learn GPU, load GPU
        #self.model = Net()
        #self.model = self.model.load_state_dict(torch.load(model_path))
        ## learn CPU, load GPU
        #self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        #self.train_optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        #self.test_optimizer = optim.SGD(self.model
        summary(self.model, [(2, 128, 128), (4,)])

    def load_depth(self):
        # imageは，subscribeではなく，最新のpickleをloadする．
        depth_path = "Data/depth_data"
        self.depth_dataset = np.empty((0,16384)) #230400))
        self.gray_dataset = np.empty((0,16384)) #230400))
        depth_key = 'heightmap_image.pkl'
        color_key = 'extract_color_image.pkl'
        t_cnt = 0
        tmp_cnt = 0
        for d_dir_name, d_sub_dirs, d_files in sorted(os.walk(depth_path), reverse=True): 
            for df in sorted(d_files, reverse=True):
                if color_key == df[-len(color_key):]:
                    with open(os.path.join(d_dir_name, df), 'rb') as f:
                        fff = pickle.load(f)
                        color_image = fff
                        WIDTH = 64#240
                        HEIGHT = 64#240
                        """
                        bridge = CvBridge()
                        try:
                            color_image = bridge.imgmsg_to_cv2(ff, 'passthrough')
                        except CvBridgeError, e:
                            rospy.logerr(e)
                        """
                        im = fff.reshape((480,640,3))
                        pil_im = Image.fromarray(np.uint8(im))
                        pil_im = pil_im.resize((129, 172))
                        im = np.asarray(pil_im)
                        im_gray = 0.299 * im[:, :, 0] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 2]
                        h, w = im_gray.shape
                        x1 = (w / 2) - WIDTH
                        x2 = (w / 2) + WIDTH
                        y1 = (h / 2) - HEIGHT
                        y2 = (h / 2) + HEIGHT
                        gray_data = np.empty((0,16384))

                        for i in range(y1, y2):
                            for j in range(x1, x2):
                                if im_gray.item(i,j) == im_gray.item(i,j):
                                    gray_data = np.append(gray_data, im_gray.item(i,j))
                                else:
                                    gray_data = np.append(gray_data, 0)
                                            
                        gray_data = np.array(gray_data).reshape((1, 16384)) #230400))

                if depth_key == df[-len(depth_key):]:
                    with open(os.path.join(d_dir_name, df), 'rb') as f:
                        ff = pickle.load(f)
                        depth_image = ff
                        WIDTH = 64#240
                        HEIGHT = 64#240
                        """
                        bridge = CvBridge()
                        try:
                            depth_image = bridge.imgmsg_to_cv2(ff, 'passthrough')
                        except CvBridgeError, e:
                            rospy.logerr(e)
                        """
                        """
                        im = ff.reshape((480,640,3))
                        im_gray = 0.299 * im[:, :, 0] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 2]
                        depth_image = im_gray
                        """
                        im = ff.reshape((128,128,2))
                        depth_image = im[:, :, 0]
                        h, w = depth_image.shape
                        x1 = (w / 2) - WIDTH
                        x2 = (w / 2) + WIDTH
                        y1 = (h / 2) - HEIGHT
                        y2 = (h / 2) + HEIGHT
                        depth_data = np.empty((0,16384)) #230400))

                        for i in range(y1, y2):
                            for j in range(x1, x2):
                                if depth_image.item(i,j) == depth_image.item(i,j):
                                    depth_data = np.append(depth_data, depth_image.item(i,j))
                                else:
                                    depth_data = np.append(depth_data, 0)
                        depth_data = np.array(depth_data).reshape((1, 16384)) #230400))
                        #self.depth_dataset = np.append(self.depth_dataset, depth_data, axis=0)
        self.depth_dataset = depth_data.reshape((1, 1, 128, 128))
        self.gray_dataset = gray_data.reshape((1, 1, 128, 128))
        self.gray_depth_dataset = np.concatenate([self.depth_dataset, self.gray_dataset], 1)
        return self.gray_depth_dataset
        print("Finished loading all depth data")
 
    def test(self, grasp_point):
        depth_data = self.load_depth()
        depth_data = depth_data.reshape(1, 2, 128, 128)
        depth_data = torch.from_numpy(depth_data).float()
        grasp_point = grasp_point.reshape(1, 4)
        grasp_point = torch.from_numpy(grasp_point).float()
        depth_data = depth_data.to(self.device)
        grasp_point = grasp_point.to(self.device)
        outputs = self.model(depth_data, grasp_point)
        labels =  torch.from_numpy(np.array(1)).float()
        # lossのgrasp_point偏微分に対してoptimaizationする．
        depth_data.requires_grad = False
        grasp_point.requires_grad = True
        loss = self.criterion(outputs.view_as(labels), labels)
        loss.backward()
        self.test_optimizer.step()
        _, inferred_grasp_point = torch.max(outputs.grasp_point, 1)

        return inferred_grasp_point
        # 最適化されたuを元に把持を実行し、その結果を予測と比較する

def inferred_point_callback(data):
    gen = point_cloud2.read_points(data, field_names = ("x", "y", "z"), skip_nans=True)
    length = 1 
    A = np.arange(3, dtype=float).reshape(1,3)
    # Whole edge (x,y,z) points
    for l in gen:
        l = np.array(l, dtype='float')
        l = l.reshape(1,3)
        A = np.append(A, l, axis=0)
        length += 1

    # Randomly choose one grasp point
    idx = np.random.randint(length, size=1) #To do : change 10 to data length
    Ax = A[idx, 0]
    Ay = A[idx, 1]
    Az = A[idx, 2]
    """
    Romdomly choose rotation theta from 0, 45, 90. (currently, 90 for test).
    Other roatation angle is fixed toward middle point.
    But currently, rotation is fixed for test. 
    """
    # euler angle will be strange when converting in eus program. Adjust parameter until solving this problem.  
    phi_list = [math.pi/2, math.pi*4/6, math.pi*5/6]
    theta = 0 #-1.54 
    phi = random.choice(phi_list) #1.2(recentry, 2.0)
    psi = 0
    random_grasp_posrot = np.array((Ax, Ay, Az, phi), dtype='float').reshape(1,1,4) #reshape(1,4) 
    
    # inference
    ts = TestSystem()
    inferred_grasp_point = ts.test(random_grasp_posrot)
    #ts.pub_inferred_grasp_point(inferred_grasp_point)
    Ax = inferred_grasp_point[0]
    Ay = inferred_grasp_point[1]
    Az = inferred_grasp_point[2]
    phi = inferred_grasp_point[3]
    if phi < 1.6:
        q_phi = 0
    else:
        q_phi = phi
    q = tf.transformations.quaternion_from_euler(theta, q_phi, psi)
    posestamped = PoseStamped()
    pose = posestamped.pose
    pose.position.x = Ax
    pose.position.y = Ay
    pose.position.z = Az 
    pose.orientation.x = q[0]
    pose.orientation.y = q[1]
    pose.orientation.z = q[2]
    pose.orientation.w = q[3]
    header = posestamped.header
    header.stamp = rospy.Time(0)
    header.frame_id = "head_mount_kinect_rgb_optical_frame"
    now = datetime.datetime.now()
    walltime = str(int(time.time()*1000000000))
    filename = 'Data/inferred_grasp_point/' + walltime + '.pkl'
    with open(filename, "wb") as f:
        pickle.dump(grasp_posrot, f)
        print("saved inferred grasp point")
    pub.publish(posestamped)

if __name__ == '__main__':
    ts = TestSystem()
    # load model and test
    ts.load_model()
    #subscribe edge pointcloud data
    try:
        rospy.init_node('grasp_point_server')
        rospy.Subscriber('/organized_edge_detector/output', PointCloud2, inferred_point_callback, queue_size=1000)
        pub = rospy.Publisher('/grasp_point', PoseStamped, queue_size=100)
        """
        while not rospy.is_shutdown():
            data = rospy.wait_for_message('supervoxel_segmentation/output/cloud', PointCloud2)
            choose_point_callback(data)
            break
        """
        rospy.spin()
    except rospy.ROSInterruptException: pass

