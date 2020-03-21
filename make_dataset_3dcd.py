"""
Author: Md Mahedi Hasan
Description: Preprocess pose sequence dataset to feed rnn model
Steps to do
        1. find out and sort partial body
        2. normalize keypoints
        3. handle no person, multiple person
        4. make train & validation dataset
"""

# python packages
import os
import json, glob
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.utils import to_categorical


# project modules
from .. import config
from . import hand_features_3dcd as hf

# for motion features
first_frame_bkps = []

# formating json file
def handling_json_data_file(data):
    global first_frame_bkps
    combined_features = []
    is_no_people = False
    is_partial_body = False
    
    # no people detected
    if len(data["people"]) == 0:
        is_no_people = True

    # one people detected 
    else:
        pose_keypoints = data["people"][0]["pose_keypoints_2d"]
        is_partial_body =  hf.is_partial_body(pose_keypoints)

        # for complete pose
        if(not is_partial_body):
            pose_features = hf.normalize_keypoints(pose_keypoints)

        """
            limb_features = hf.get_body_limb(pose_keypoints)
            angle_features = hf.get_joint_angle(pose_keypoints)
            
            # for first frame, store the  bpks and skip the motion feat.
            if(len(first_frame_bkps) == 0):
                first_frame_bkps = pose_keypoints
                is_no_people = True

            else:
                second_frame_bpks = pose_keypoints
                motion_features = hf.get_motion_featurs(second_frame_bpks, 
                                                  first_frame_bkps)
                first_frame_bkps = second_frame_bpks
        """
    
    # combining all fetures
    if (not is_partial_body and not is_no_people):
        combined_features = pose_features
    """
        combined_features += limb_features
        combined_features += angle_features
        combined_features += motion_features
    """
    return combined_features, is_no_people, is_partial_body



def get_format_data(cheat_vid_data, cheat_vid_label):
    nb_images = len(cheat_vid_data)
    success = False

    if (nb_images >= config.nb_steps): 
        success = True
        d =  np.array(cheat_vid_data[:config.nb_steps])
        l = cheat_vid_label[:config.nb_steps]
        return d, l, success

    else:
        return cheat_vid_data, cheat_vid_label, success



def get_keypoints_for_all_cheat(cheat_type_list):

    print("\n\n*********** Generating %s data ***********" % "training")    
    train_dataset = []
    valid_dataset = []

    train_dataset_label = []
    valid_dataset_label = []

    for ctype, cheat_type in enumerate (cheat_type_list):
        print("\n\n\n\n############ cheat type %s ############" % cheat_type)

        # variable for each cheat type
        cheat_label = config.cheat_lable_list.index(cheat_type)
        

        # getting angle
        cheat_dir = os.path.join(config.data_3dcd_path(), cheat_type)
        cheat_vid_list = os.listdir(cheat_dir)
        #print(cheat_vid_list)

        num_cheat_vid =  len(cheat_vid_list)
        print("%s has: %d cheat vidoes" % (cheat_type, num_cheat_vid))

        missing_video = 0
        type_dataset = []
        type_label = []
        
        # considering each cheat video
        for cheat_vid in cheat_vid_list:
            cheat_vid_dir = os.path.join(cheat_dir, cheat_vid)

            # considering each cheat vids
            cheat_vid_data = []
            cheat_vid_label = []
            is_missing_frame = False
            os.chdir(cheat_vid_dir)

            # getting all json files
            json_files = sorted(glob.glob("*.json"))

            missing_count = 0
            for f in (json_files): 
                with open(f) as data_file:
                    data = json.load(data_file)
                    
                    #print("new frame")
                    frame_kps, no_people, partial_body = handling_json_data_file(data)
                    #print(frame_kps)
        
                    # counting no and partial body detected
                    if (no_people == True or partial_body == True):  
                        
                        missing_count += 1
                        is_missing_frame = True 
                        
                    # for single people save the frame key points
                    else:
                        cheat_vid_data.append(frame_kps)
                        cheat_vid_label.append(cheat_label)
            
            #print(len(cheat_vid_data))
            
            # count total misssing videos
            vid_data, vid_label, success = get_format_data(cheat_vid_data, cheat_vid_label)
            if(success == True):
                type_dataset.append(np.expand_dims(vid_data, axis=0))
                type_label.append(vid_label)
            
            else:
                missing_video += 1

        print("for label: ", cheat_label, "total missing: ", missing_video)


        # validation 15%
        nb_valid_vid = int((num_cheat_vid - missing_video) * 0.15)

        if(ctype == 0):
            valid_dataset = np.vstack(type_dataset[:nb_valid_vid])
            train_dataset = np.vstack(type_dataset[nb_valid_vid:])

            valid_dataset_label = to_categorical(np.array(
                        type_label[:nb_valid_vid]), config.nb_classes)

            train_dataset_label = to_categorical(np.array(
                        type_label[nb_valid_vid:]), config.nb_classes)

        else:
            valid_dataset = np.concatenate((valid_dataset, 
                            np.vstack(type_dataset[:nb_valid_vid])), axis = 0) 

            train_dataset = np.concatenate((train_dataset, 
                            np.vstack(type_dataset[nb_valid_vid:])), axis = 0)

            valid_dataset_label = np.concatenate((valid_dataset_label, 
                        to_categorical(np.array(type_label[:nb_valid_vid]), 
                        config.nb_classes)), axis = 0)

            train_dataset_label = np.concatenate((train_dataset_label, 
                        to_categorical(np.array(type_label[nb_valid_vid:]), 
                        config.nb_classes)), axis = 0)


    print("train dataset shape: ", train_dataset.shape)
    print("valid dataset shape", valid_dataset.shape)

    print("train label shape: ", train_dataset_label.shape)
    print("valid label shape: ", valid_dataset_label.shape)
    
    return train_dataset, valid_dataset, train_dataset_label, valid_dataset_label



def get_train_data():
    cheat_type_list = os.listdir(config.data_3dcd_path())
    cheat_type_list = sorted(cheat_type_list)

    return get_keypoints_for_all_cheat(cheat_type_list)


if __name__ == "__main__":
    get_train_data()