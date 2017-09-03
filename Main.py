'''
Created on 2017/09/04

@author: Abe
'''
from DeepLearningSoundData import DeepLearningSoundData
from TesorFlow import TesorFlow


if __name__ == '__main__':
    
    sub_tr_dirs = ["fold1"]
    sub_ts_dirs = ["fold2"]
    dlSoundData = DeepLearningSoundData(sub_tr_dirs, sub_ts_dirs)
    tr_features, tr_labels, ts_features, ts_labels, lable_num = dlSoundData.create_sound_data()
    
    print (tr_features)
    print (tr_labels)
    print (ts_features)
    print (ts_labels)
    print (lable_num)
    
    tensol = TesorFlow(tr_features, tr_labels, ts_features, ts_labels, lable_num)
    tensol.execute()