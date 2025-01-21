import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction
from sentence_transformers import SentenceTransformer


class get_model(nn.Module):
    def __init__(self, output_size=7, normal_channel=False):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1408, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, output_size)

        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2') # Embedding size is 384
        # Freeze the text encoder parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def forward(self, xyz, text):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        print("l1_points.shape: ", l1_points.shape)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        print("l2_points.shape: ", l2_points.shape)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        print("l3_points.shape: ", l3_points.shape)
        
        text_embedding = self.text_encoder.encode(text, convert_to_tensor=True)
        # Ensure gradients are not calculated for the embeddings
        text_embedding.requires_grad = False
        pcd_embedding = l3_points.view(B, 1024)

        # Concatenate PointNet++ Output and Text Embedding
        x = torch.cat((pcd_embedding, text_embedding), dim=1)  # Shape: (batch_size, 1024 + text_embedding_dim)
        print("x.shape: ", x.shape)


        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        # x = F.log_softmax(x, -1)


        return x, l3_points



class get_loss(torch.nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def quaternion_loss(self, q_pred, q_true):
        # Normalize quaternions
        q_pred = F.normalize(q_pred, dim=-1)
        q_true = F.normalize(q_true, dim=-1)
        
        # Angular distance loss
        loss = 1.0 - torch.abs(torch.sum(q_pred * q_true, dim=-1))  # Batch-wise dot product
        return loss.mean()
    
    def forward(self, pred, target, lambda_pos=1.0, lambda_orn=1.0):
        print("pred, target: ", pred.shape, target.shape)
        pos_loss = F.mse_loss(pred[:, :3], target[:, :3])

        # Orientation loss
        orn_loss = self.quaternion_loss(pred[:, 3:], target[:, 3:])
        
        # Combined loss
        total_loss = lambda_pos * pos_loss + lambda_orn * orn_loss
        # print("total_loss: ", total_loss)
        loss_dict = {'total': total_loss}
        return loss_dict
        # # total_loss = self.axis_loss_scale * axis_loss 
        # loss_dict = {'total': total_loss, 
        #              'axis': axis_loss, 
        #              'mat_diff': mat_diff_loss}
        # return loss_dict