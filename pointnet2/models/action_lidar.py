import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2.models.pointnet2_utils import PointNetSetAbstraction


class get_model(nn.Module):
    def __init__(self, lidar_input_size=1090, action_input_size=3):
        super(get_model, self).__init__()

        # LiDAR input processing with 1D convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        
        # Fully connected layers for LiDAR feature extraction
        self.fc_lidar = nn.Linear(lidar_input_size * 64, 128)
        self.lidar_bn1 = nn.BatchNorm1d(128)
        self.lidar_drop1 = nn.Dropout(0.2)

        # Fully connected layers for action input
        # self.fc_action = nn.Linear(action_input_size, 128)
        
        self.action_fc1 = nn.Linear(action_input_size, 64)
        self.action_bn1 = nn.BatchNorm1d(64)
        self.action_drop1 = nn.Dropout(0.2)
        self.action_fc2 = nn.Linear(64, 128)
        self.action_bn2 = nn.BatchNorm1d(128)
        self.action_drop2 = nn.Dropout(0.2)

        # Fusion and classification layers
        self.fc_fusion = nn.Linear(128 + 128, 64)
        self.fc_fusion_bn1 = nn.BatchNorm1d(64)
        self.fc_fusion_drop1 = nn.Dropout(0.2)
        self.fc_out = nn.Linear(64, 1)


    def forward(self, lidar, action):
        # Process LiDAR input
        x = lidar.unsqueeze(1)  # Add channel dimension for Conv1d
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        # x = F.relu(self.fc_lidar(x))
        x = self.lidar_drop1(F.relu(self.lidar_bn1(self.fc_lidar(x))))

        # Process action input
        # a = F.relu(self.fc_action(action))
        a = self.action_drop1(F.relu(self.action_bn1(self.action_fc1(action))))
        a = self.action_drop2(F.relu(self.action_bn2(self.action_fc2(a))))

        # Fuse features
        combined = torch.cat((x, a), dim=1)
        # combined = F.relu(self.fc_fusion(combined))
        combined = self.fc_fusion_drop1(F.relu(self.fc_fusion_bn1(self.fc_fusion(combined))))

        # Output layer
        output = self.fc_out(combined)

        return output
    
    # def forward(self, lidar, action):
    #     # Process LiDAR input
    #     x = lidar.unsqueeze(1)  # Add channel dimension for Conv1d
    #     x = F.relu(self.conv1(x))
    #     x = F.relu(self.conv2(x))
    #     x = F.relu(self.conv3(x))
    #     x = x.view(x.size(0), -1)  # Flatten
    #     x = F.relu(self.fc_lidar(x))

    #     # Process action input
    #     a = F.relu(self.fc_action(action))

    #     # Fuse features
    #     combined = torch.cat((x, a), dim=1)
    #     combined = F.relu(self.fc_fusion(combined))

    #     # Output layer
    #     output = self.fc_out(combined)

    #     return output


class get_loss(nn.Module):
    def __init__(self, num_zeros, num_ones):
        super(get_loss, self).__init__()
        weight = num_zeros / num_ones
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight))


    def forward(self, pred, target):
        # print("pred: ", pred)
        # print("-----")
        # print("target: ", target)
        total_loss = self.criterion(pred, target)
        # print("total_loss: ", total_loss)

        return total_loss


# Example usage
if __name__ == "__main__":
    lidar_input_size = 360  # Example: 360 LiDAR points
    batch_size = 8
    
    model = get_model(lidar_input_size=lidar_input_size)

    # Random example inputs
    lidar_data = torch.rand(batch_size, lidar_input_size)
    action_data = torch.rand(batch_size, 3)  # Example: (x, y, yaw)

    # Forward pass
    collision_prediction = model(lidar_data, action_data)
    print(collision_prediction)