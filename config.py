"""
Author: 
Description: this file contains configuration info for temporal network
"""

# python packages
import os

cheat_lable_list = ['0_Nocheat', '1_LookLeft', '2_LookRight', '3_PocketSheet', 
                    '4_PantsSheet', '5_ExchPaper', '6_FaceCodes', '7_HandCodes']

nb_classes = 8
shape = (31, 24, 1)
lr = 0.001
batch_size = 200
nb_epochs = 60

def root_path():
    return os.path.dirname(os.path.abspath(__file__))


def checkpoint_path():
    return os.path.join(root_path(), "checkpoint")


def openpose_path():
    return os.path.join(root_path(), "libs", "openpose")


def dataset_3dcd_path():
    return os.path.join(root_path(), "dataset", "Data3DCD")


def data_3dcd_path():
    return os.path.join(root_path(), "data", "Data3DCD", "frames")


def pose_3dcd_path():
    return os.path.join(root_path(), "data", "Data3DCD", "pose")


def checkpoint_path():
    return os.path.join(root_path(), "checkpoint")


def output_path():
    return os.path.join(root_path(), "output")



if __name__ == "__main__":
    print(dataset_3dcd_path())