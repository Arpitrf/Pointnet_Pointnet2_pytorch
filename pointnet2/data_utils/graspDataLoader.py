# *_*coding:utf-8 *_*
import os
import json
import warnings
import open3d as o3d
from pickle import load
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')
import provider
from scipy.spatial.transform import Rotation as R

f_name_to_task = {
    "pringles_place_in_shelf": "place in shelf",
    "pringles_place_in_box": "place in box",
    "open_drawer": "open drawer",
    "open_fridge": "open fridge"
}

def pc_normalize(pc, axis=None):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    if axis is not None:
        axis = axis - centroid
        axis = axis / m
        return pc, axis
    else:
        return pc

class GraspDataset(Dataset):
    def __init__(self, root = './data', split='train', npoints=2500, normal_channel=False):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.normal_channel = normal_channel
        self.datapts = []

        f_names = ["pringles_place_in_shelf", "pringles_place_in_box"]
        for f_name in f_names:
            object_name = f_name.split("_")[0]
            print("object_name: ", object_name)
            f = open(f"{self.root}/{object_name}_pcd.pkl", 'rb')
            pcd_dict = load(f)
            pcd = np.concatenate((pcd_dict["points"], pcd_dict["normals"]), axis=1)
            if not self.normal_channel:
                pcd = pcd[:, 0:3]
            
            f = open(f"{self.root}/{f_name}.pkl", 'rb')
            T_grasp = load(f)
            grasp_pos = T_grasp[:3, 3]
            grasp_quat = R.from_matrix(T_grasp[:3, :3]).as_quat()
            grasp_pose = np.concatenate((grasp_pos, grasp_quat))
            print("grasp_pose: ", grasp_pose)
            
            datapt = {
                "text_description": f_name_to_task[f_name],
                "grasp": grasp_pose,
                "pcd": pcd
            }
            self.datapts.append(datapt)
        
        # breakpoint()

        # # normalize the pcd and also the q (s_hat is already a unit vector so no need to normalize it)
        # # print("--before: ", self.axis[1])
        # self.point_set[:, 0:3], self.axis[1] = pc_normalize(self.point_set[:, 0:3], self.axis[1])
        # # print("--after: ", self.axis[1])

        # set augmentation ranges
        self.disp_range = [0.2, 0.2, 0.2]  # 20 cm
        self.angle_range = 0.78  # 45 degrees

    def __getitem__(self, idx):

        # chosse a datapt
        datapt = np.random.choice(self.datapts)
        pcl = datapt["pcd"]
        text_description = datapt["text_description"]
        grasp = datapt["grasp"]

        NUM_POINT = pcl.shape[0]
        org_colors = np.tile([0.0, 0.0, 1.0], (NUM_POINT, 1)) # blues
        o3d_pcl = o3d.geometry.PointCloud()
        o3d_pcl.points = o3d.utility.Vector3dVector(pcl[:, :3])
        o3d_pcl.colors = o3d.utility.Vector3dVector(org_colors)
        o3d.visualization.draw_geometries([o3d_pcl])

        # translation
        if self.split == 'train':
            # uniform random sampling
            disp_x = np.random.uniform(-self.disp_range[0], self.disp_range[0])
            disp_y = np.random.uniform(-self.disp_range[1], self.disp_range[1])
            disp_z = np.random.uniform(-self.disp_range[2], self.disp_range[2])
        elif self.split == 'val':
            # fixed values based on idx
            disp_x = -self.disp_range[0] + 2*self.disp_range[0]*idx/self.__len__()
            disp_y = -self.disp_range[1] + 2*self.disp_range[1]*idx/self.__len__()
            disp_z = 0.0
            # print("disp_x, disp_y,: ", disp_x, disp_y)
        disp = np.array([disp_x, disp_y, disp_z])
        pcl[:, :3], grasp = self.translation_augmentation(pcl, grasp, disp)

        # # rotation
        # if self.split == 'train':
        #     # uniform random sampling
        #     self.angle_radians = np.random.uniform(-self.angle_range, self.angle_range)
        # elif self.split == 'val':
        #     # fixed values based on idx
        #     self.angle_radians = -self.angle_range + 2*self.angle_range*idx/self.__len__()
        # pcl[:,:3], axis = self.rotation_augmentation(pcl[:,:3], axis, self.angle_radians)

        # print("translation and rotation aug applied: ", disp, self.angle_radians)
        # if self.split == 'train':
        #     visualize_pcl_axis([axis], pcl.shape[0], pcl[:, :3], savepath='/home/arpit/test_projects/bimanual_predictor/temp.png', use_q=self.use_q)

        # random scaling
        rand_val = np.random.uniform(0.0, 1.0)
        if rand_val > 0.75:
            pcl_temp = np.expand_dims(pcl, axis=0)
            pcl_temp[:, :, 0:3] = provider.random_scale_point_cloud(pcl_temp[:, :, 0:3])
            pcl = pcl_temp[0]
 
        # Obtain normals
        if self.normal_channel:
            pcl_o3d = o3d.geometry.PointCloud()
            pcl_o3d.points = o3d.utility.Vector3dVector(pcl[:,:3])
            pcl_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            pcl[:, 3:6] = np.array(pcl_o3d.normals)

        return pcl, grasp, text_description

    
    def __len__(self):
        if self.split == 'train':
            return 16
        elif self.split == 'val':
            return 16
    
    def translation_augmentation(self, points, grasp, disp=np.array([0.05, 0.05, 0.0])):
        translation_matrix = np.array([
            [1, 0, 0, disp[0]],
            [0, 1, 0, disp[1]],
            [0, 0, 1, disp[2]],
            [0, 0, 0, 1]
        ])
        # Apply the translation to the point cloud
        # making dimenson (n,3) -> (n,4) by appending 1 to each point
        ones_column = np.ones((points.shape[0], 1), dtype=points.dtype)
        points = np.append(points, ones_column, axis=1)
        transformed_points = np.dot(points, translation_matrix.T)
        
        # Apply the translation to the grasp
        T_grasp = np.eye(4)
        T_grasp[:3, 3] = grasp[:3]
        T_grasp[:3, :3] = R.from_quat(grasp[3:]).as_matrix()
        T_transformed_grasp = np.dot(T_grasp, translation_matrix.T)
        grasp_pos = T_transformed_grasp[:3, 3]
        grasp_quat = R.from_matrix(T_transformed_grasp[:3, :3]).as_quat()
        transformed_grasp_pose = np.concatenate((grasp_pos, grasp_quat))

        return transformed_points[:, :3], transformed_grasp_pose

    def rotation_augmentation(self, points, axis, angle_radians=0.1):
        # Define the 3D rotation matrix around the z-axis
        rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                                    [np.sin(angle_radians), np.cos(angle_radians), 0],
                                    [0, 0, 1]])
        # Apply the rotation to the point cloud
        transformed_points = np.dot(points, rotation_matrix.T)
        # Apply the rotation to the gt
        transformed_axis = np.dot(axis, rotation_matrix.T)
        # print("transformed_axis norm: ", transformed_axis.shape, np.linalg.norm(transformed_axis))
        transformed_axis[0] /= np.linalg.norm(transformed_axis[0])
        # transformed_axis /= np.linalg.norm(transformed_axis)
        
        return transformed_points, transformed_axis