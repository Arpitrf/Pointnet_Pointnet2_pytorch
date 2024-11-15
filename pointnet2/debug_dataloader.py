import torch

from data_utils.OGDataLoader import SequenceDataset, prepare_data


dataset = SequenceDataset(
    hdf5_path='/home/arpit/test_projects/OmniGibson/place_in_shelf_data/dataset.hdf5',
    obs_keys=('pcd',),  # observations we want to appear in batches
    # obs_info_keys=('seg_instance_id_info',),
    dataset_keys=(  # can optionally specify more keys here if they should appear in batches
        "actions",
        "grasped",
        "contacts"
    ),
    seq_length=1,  # length-10 temporal sequences
    pad_seq_length=True,  # pad last obs per trajectory to ensure all sequences are sampled
    hdf5_normalize_obs=False,
    filter_by_attribute='train',  # filter either train or validation data
    image_size=[64, 64],
)
print("type(dataset): ", type(dataset))
print("len(dataset): ", len(dataset))
print("dataset.n_demos: ", dataset.n_demos)

datapoint = dataset[10]
print("actions: ", datapoint['actions'].shape)
print("grasped: ", datapoint['grasped'].shape)
print("contact: ", datapoint['contacts'].shape)
print("keys: ", datapoint['obs'].keys())
print("pcd_points: ", datapoint['obs']['pcd_points'].shape)
print("pcd_colors: ", datapoint['obs']['pcd_colors'].shape)
print("pcd_normals: ", datapoint['obs']['pcd_normals'].shape)

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
print("batch[actions], batch[points], batch[contacts]: ", batch['actions'].shape, batch['points'].shape, batch['contacts'].shape)