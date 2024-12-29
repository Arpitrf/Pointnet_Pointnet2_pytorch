import torch
import numpy as np

from data_utils.OGDataLoader import SequenceDataset, prepare_data

target_key = "contacts"
dataset = SequenceDataset(
    hdf5_path='/home/arpit/projects/Pointnet_Pointnet2_pytorch/place_in_shelf/dataset.hdf5',
    obs_keys=('pcd',),  # observations we want to appear in batches
    # obs_info_keys=('seg_instance_id_info',),
    dataset_keys=(  # can optionally specify more keys here if they should appear in batches
        "actions",
        target_key,
    ),
    target_key = target_key, 
    seq_length=1,  # length-10 temporal sequences
    pad_seq_length=True,  # pad last obs per trajectory to ensure all sequences are sampled
    hdf5_normalize_obs=False,
    filter_by_attribute='train',  # filter either train or validation data
    image_size=[64, 64],
)
print("type(dataset): ", type(dataset))
print("len(dataset): ", len(dataset))
print("dataset.n_demos: ", dataset.n_demos)

for i in range(150, 200):
    datapoint = dataset[i]
    # print("actions: ", datapoint['actions'])
    # print("labels: ", datapoint['labels'])
    # # print("grasps: ", datapoint['grasps'].shape)
    # # print("contact: ", datapoint['contacts'].shape)
    # print("keys: ", datapoint['obs'].keys())
    # print("pcd_points: ", datapoint['obs']['pcd_points'].shape)
    # print("pcd_colors: ", datapoint['obs']['pcd_colors'].shape)
    # print("pcd_normals: ", datapoint['obs']['pcd_normals'].shape)

    print(np.isnan(datapoint['obs']['pcd_points']).any())

breakpoint()

trainDataLoader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=None,  # no custom sampling logic (uniform sampling)
        batch_size=24,
        shuffle=True, 
        num_workers=1,  
        # drop_last=True,  # don't provide last batch in dataset pass if it's less than 100 in size
        collate_fn=prepare_data,
    )

batch = next(iter(trainDataLoader))
print("batch: ", batch.keys())
print("batch[actions], batch[points], batch[labels]: ", batch['actions'].shape, batch['points'].shape, batch['labels'].shape)
