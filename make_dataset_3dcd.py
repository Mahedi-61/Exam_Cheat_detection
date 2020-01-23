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




def get_keypoints_for_all_cheat(cheat_type_list):

    print("\n\n*********** Generating %s data ***********" % "training")    
    total_dataset = []
    total_dataset_label = []

    for cheat_type in cheat_type_list:
        print("\n\n\n\n############ cheat type %s ############" % cheat_type)

        # variable for each cheat type
        cheat_label = config.cheat_lable_list.index(cheat_type)
        

        # getting angle
        cheat_dir = os.path.join(config.pose_3dcd_path(), cheat_type)
        cheat_vid_list = os.listdir(cheat_dir)
        #print(cheat_vid_list)

        num_cheat_vid =  len(cheat_vid_list)
        print("%s has: %d cheat vidoes" % (cheat_type, num_cheat_vid))


        missing_video = 0
        
        # considering each cheat video
        for cheat_vid in cheat_vid_list:
            cheat_vid_dir = os.path.join(cheat_dir, cheat_vid)

            cheat_vid_data = []
            is_missing_frame = False
            
            # considering each cheat vids
            os.chdir(cheat_vid_dir)

            # getting all json files
            json_files = sorted(glob.glob("*.json"))

            for f in (json_files): 
                with open(f) as data_file:
                    data = json.load(data_file)
                    
                    frame_kps, no_people, partial_body = handling_json_data_file(data)
                    #print("frame no: ", f+1); print(frame_kps)
        
                    # counting no, multiple people and partial body detected
                    if (no_people == True or partial_body == True):  
                        is_missing_frame = True 
                        
                    # for single people save the frame key points
                    else:
                        cheat_vid_data.append(frame_kps)
                        

            # count total misssing videos
            if (is_missing_frame == True):   missing_video += 1

            # appending non-missing videos
            if(is_missing_frame == False):
                total_dataset.append(np.array(cheat_vid_data))
                total_dataset_label.append(cheat_label)

        print("total missing: ", missing_video, "for label: ", cheat_label)
    
    # forming final data and label
    for  i, array in enumerate(total_dataset):
        #print(array.shape)
        if (array.shape[0] != 31):
            print("culprit: ", i)
            del total_dataset_label[i]

        else:
            if (i == 0): data = np.expand_dims(array, axis=0)
            else: data = np.concatenate((data, np.expand_dims(array, axis=0)), axis = 0)


    # on-hot encoding
    total_dataset_label = np.array(total_dataset_label)
    total_dataset_label = to_categorical(total_dataset_label, config.nb_classes)
    
    data = np.expand_dims(data, axis = 3)

    print("dataset shape: ", data.shape)
    print("label shape: ", total_dataset_label.shape)
    
    return data, total_dataset_label



def get_train_data():
    cheat_type_list = os.listdir(config.pose_3dcd_path())
    cheat_type_list = sorted(cheat_type_list)

    return get_keypoints_for_all_cheat(cheat_type_list)



if __name__ == "__main__":
    get_train_data()