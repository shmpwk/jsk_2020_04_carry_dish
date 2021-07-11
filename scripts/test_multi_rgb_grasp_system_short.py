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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import NearestNeighbors 
import seaborn as sns
sns.set_style("darkgrid")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # dynamics-net (icra2019の紐とか柔軟物を操るやつ) by Mr. Kawaharazuka
        self.conv1 = nn.Conv2d(3, 4, 3, 2, 1)
        self.cbn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, 3, 2, 1)
        self.cbn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, 3, 2, 1)
        self.cbn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, 3, 2, 1)
        self.cbn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 64, 3, 2, 1)
        self.cbn5 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8 + 4, 12)
        self.fc6 = nn.Linear(12, 1) # output is 1 dim scalar probability

    # depth encording without concate grasp point
    def forward(self, x, y):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = self.cbn1(x)
        #x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv2(x))
        x = self.cbn2(x)
        x = F.relu(self.conv3(x))
        x = self.cbn3(x)
        x = F.relu(self.conv4(x))
        x = self.cbn4(x)
        x = F.relu(self.conv5(x))
        x = self.cbn5(x)
        x = x.view(-1, self.num_flat_features(x))
        #self.depth_data =self.depth_data.view(self.depth_data.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        z = torch.cat((x, y), dim=1)
        z = F.relu(self.fc5(z))
        z = self.fc6(z)
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
        model_path = 'Data/trained_model/model_20201130_134113.pth'
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
        summary(self.model, [(3, 128, 128), (4,)])

    def load_depth(self):
        # imageは，subscribeではなく，最新のpickleをloadする．
 
        depth_path = "Data/test_depth_data"
        #self.self.depth_dataset = np.empty((0,16384)) #230400))
        #self.self.depth_dataset = np.empty((0,16384*3)) #230400))
        depth_key = 'heightmap_image.png'
        color_key = 'extract_color_image.pkl'
        tmp_cnt = 0
        for d_dir_name, d_sub_dirs, d_files in sorted(os.walk(depth_path)): 
            for df in sorted(d_files):
                print("df", df)
                print(d_dir_name)
                if depth_key == df[-len(depth_key):]:
                    with open(os.path.join(d_dir_name, df), 'rb') as f:
                        #im = pickle.load(f)
                        im = Image.open(f)
                        WIDTH = 64#240
                        HEIGHT = 64#240
                        #im = im.reshape((128,128,2)) #for pkl!
                        #depth_image = im[:, :, 0]
                        #h, w = depth_image.shape
                        h, w = im.size
                        depth_image = im
                        #h = im.shape[0]
                        #w = im.shape[1]
                        x1 = (w / 2) - WIDTH
                        x2 = (w / 2) + WIDTH
                        y1 = (h / 2) - HEIGHT
                        y2 = (h / 2) + HEIGHT
                        #self.depth_data = np.empty((0,16384)) #230400))
                        self.depth_data = np.empty((0,16384*3)) #230400))
                        for i in range(y1, y2):
                            for j in range(x1, x2):
                                if depth_image.getpixel((i,j)) == depth_image.getpixel((i,j)):
                                    self.depth_data = np.append(self.depth_data, depth_image.getpixel((i,j)))
                                else:
                                    self.depth_data = np.append(self.depth_data, 0)

                        #self.depth_data = np.array(self.depth_data).reshape((1, 16384)) #230400))
                        self.depth_data = np.array(self.depth_data).reshape((1, 3*16384)) #230400))
                        #self.self.depth_dataset = np.append(self.self.depth_dataset, self.depth_data, axis=0)
        #self.self.depth_dataset = self.self.depth_dataset.reshape((1600, 1, 480, 480))
        #self.self.depth_dataset = self.self.depth_dataset.reshape((1, 3, 128, 128))
        self.depth_data = self.depth_data.reshape((1, 3, 128, 128))
        #self.self.depth_dataset = self.self.depth_dataset.reshape((1630, 3, 128, 128))
        #rotimg = RotateImage(self.self.depth_dataset)
        #self.self.depth_dataset = np.array(rotimg.calc())
        print("Finished loading all depth data")
        return self.depth_data
 
    def test(self, grasp_point):
        self.depth_data = self.load_depth()
        self.depth_data = self.depth_data.reshape(1, 3, 128, 128)
        self.depth_data = torch.from_numpy(self.depth_data).float()
        grasp_point = grasp_point.reshape(1, 4)
        grasp_point = torch.from_numpy(grasp_point).float()
        self.depth_data = self.depth_data.to(self.device)
        grasp_point = grasp_point.to(self.device)
        self.depth_data.requires_grad = False
        grasp_point.requires_grad = True
        outputs = self.model(self.depth_data, grasp_point)
        labels =  torch.from_numpy(np.array(1)).float()
        # lossのgrasp_point偏微分に対してoptimaizationする．
        loss = self.criterion(outputs.view_as(labels), labels)
        loss.backward()
        self.test_optimizer.step()
        #_, inferred_grasp_point = torch.max(outputs.grasp_point, 1)
        gamma = 1
        print("grad", grasp_point.grad)
        inferred_grasp_point = grasp_point - gamma * grasp_point.grad
        inferred_grasp_point = inferred_grasp_point.to('cpu').detach().numpy().copy()
        print("inferred_grasp_point", inferred_grasp_point)
        return inferred_grasp_point
        # 最適化されたuを元に把持を実行し、その結果を予測と比較する

class InflateGraspPoint(object):
    def __init__(self, grasp_point):
        self.aug_grasp_point = np.zeros((10, 4))
        self.x = grasp_point[[0]]
        self.y = grasp_point[[1]]
        self.z = grasp_point[[2]]
        self.theta = grasp_point[[3]]
        self.times = 8

    def calc(self):
        r = math.sqrt(self.x**2 + self.y**2)
        rad = math.atan2(self.y, self.x)
        rad_delta = math.pi/(self.times/2)
        X = np.zeros(self.times)
        Y = np.zeros(self.times)
        for i in range(self.times):
            X[i] = r * math.cos(rad + rad_delta*i)
            Y[i] = r * math.sin(rad + rad_delta*i)
            self.aug_grasp_point[i, :] = np.array((X[i], Y[i], self.z, self.theta))
        return self.aug_grasp_point

def nearest_point(points, inferred_point):
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.plot(points[:,0], points[:,1], points[:,2], marker='o', linestyle='None') 
    #plt.show()
    """
    test_datapoint = inferred_point.tolist()
    #test_datapoint = [4.3, 2.7, 4.2]
    k = 1
    knn_model = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(points)
    distances, indices = knn_model.kneighbors([test_datapoint])
    #print("K Nearest Neighbors:")
    for rank, index in enumerate(indices[0][:k], start=1):
        print(points[index])
    """ 
    ax.plot(points[indices][0][:][:, 0], points[indices][0][:][:, 1], points[indices][0][:][:, 2], marker='o', color='k') 
    ax.plot([test_datapoint[0]], [test_datapoint[1]], [test_datapoint[2]], marker='x', color='k')
    #plt.show()
    """
    return points[index] 

def inferred_point_callback(data):
    gen = point_cloud2.read_points(data, field_names = ("x", "y", "z"), skip_nans=True)
    length = 1 
    A = np.empty(3, dtype=float).reshape(1,3)
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
    #phi_list = [math.pi/2, math.pi*4/6, math.pi*5/6]
    phi_list = [math.pi/2, math.pi*4/6, math.pi*5/6]
    theta = 0 #-1.54 
    phi = random.choice(phi_list) #1.2(recentry, 2.0)
    psi = 0
    random_grasp_posrot = np.array((Ax, Ay, Az, phi), dtype='float').reshape(1,1,4) #reshape(1,4) 
    print("random grasp posrost", random_grasp_posrot) 
    # Inference!
    ts = TestSystem()
    inferred_grasp_point = ts.test(random_grasp_posrot)
    inferred_grasp_point = inferred_grasp_point.reshape(4)
    # Find nearest edge point based on inferred point
    nearest = nearest_point(A, inferred_grasp_point[:3:])
    Ax = nearest[0]
    Ay = nearest[1]
    Az = nearest[2]
    phi = inferred_grasp_point[3] - 0.5
    # When converting from here to eus, 
    """
    if phi < 1.6:
        q_phi = 0
    else:
        q_phi = phi
    """
    if (phi < 1.6):
        phi = 1.6
    elif (phi > 2.6):
        phi = 2.6
    q = tf.transformations.quaternion_from_euler(theta, phi, psi)
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
    # grasp_posrot is actual grasp point
    grasp_posrot = np.array((Ax, Ay, Az, phi), dtype='float').reshape(1,4) 
    print("nearest_inferred_grasp_point", grasp_posrot)
    # inferred_posrot is a inferred point but not for grasping
    inferred_posrot = np.array((inferred_grasp_point[0], inferred_grasp_point[1], inferred_grasp_point[2], inferred_grasp_point[3])) 
    
    # Save grasp_posrot and inferred_posrot
    now = datetime.datetime.now()
    walltime = str(int(time.time()*1000000000))
    filename = 'Data/inferred_grasp_point/' + walltime + '.pkl'
    with open(filename, "wb") as f:
        
        pickle.dump(grasp_posrot, f)
        print("saved inferred grasp point")
    filename = 'Data/inferred_point/' + walltime + '.pkl'
    with open(filename, "wb") as ff:
        pickle.dump(inferred_posrot, ff)
        print("saved inferred point")
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

