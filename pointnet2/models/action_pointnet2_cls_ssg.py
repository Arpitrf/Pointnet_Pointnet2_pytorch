import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2.models.pointnet2_utils import PointNetSetAbstraction


class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        # self.fc3 = nn.Linear(256, num_class)

        # action head
        # 10 for only actions and 16 for actions and eef pose
        # self.action_fc1 = nn.Linear(10, 64)
        self.action_fc1 = nn.Linear(16, 64)

        self.action_bn1 = nn.BatchNorm1d(64)
        self.action_drop1 = nn.Dropout(0.2)
        self.action_fc2 = nn.Linear(64, 32)
        self.action_bn2 = nn.BatchNorm1d(32)
        self.action_drop2 = nn.Dropout(0.2)
        # self.action_fc3 = nn.Linear(32, 16)

        # output head
        self.head_fc1 = nn.Linear(288, 64)
        self.head_bn1 = nn.BatchNorm1d(64)
        self.head_drop1 = nn.Dropout(0.4)
        self.head_fc2 = nn.Linear(64, 32)
        self.head_bn2 = nn.BatchNorm1d(32)
        self.head_drop2 = nn.Dropout(0.2)
        self.head_fc3 = nn.Linear(32, num_class)

    def forward(self, xyz, action):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        # x = self.fc3(x)

        # pass action through action head
        a = self.action_drop1(F.relu(self.action_bn1(self.action_fc1(action))))
        a = self.action_drop2(F.relu(self.action_bn2(self.action_fc2(a))))

        # concatenate the action feature and the pointnet feature
        concatenated = torch.cat((x, a), dim=1)

        c = self.head_drop1(F.relu(self.head_bn1(self.head_fc1(concatenated))))
        c = self.head_drop2(F.relu(self.head_bn2(self.head_fc2(c))))
        c = self.head_fc3(c)
        
        # Not performing sigmoid here cause using BCEwithLogits Loss which performs sigmoid internally. 
        # Remember to explicitly apply sigmoid during testing though
        # c = F.sigmoid(self.head_fc3(c)) 

        return c, l3_points



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()


    def forward(self, pred, target, trans_feat):
        # total_loss = F.nll_loss(pred, target)
        total_loss = self.criterion(pred, target)
        # print("total_loss: ", total_loss)

        return total_loss
