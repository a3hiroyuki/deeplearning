'''
Created on 2018/05/29

@author: hiroy
'''
import pandas as pd
import os
import shutil


cur_dir1 = 'C:\pleiades\workspace1\Test2'
cur_dir2 = 'C:\pleiades\workspace1\Test2\img\\'

df = pd.read_csv(cur_dir1 + '\input.txt', dtype = 'object')
print(df)

def get_label(names, ext):
    try:
        df_exact = df[df['no'] == names[0]]
        df_exact = df_exact.iloc[0]
        label = '{0}_{1}_{2}_{3}_{4}'.format(names[0], df_exact['X'], df_exact['Y'], names[1], names[2])
        return label + ext
    except:
        print("error")
        return None

print (cur_dir1)

files = os.listdir(cur_dir2)
files_file = [f for f in files if os.path.isfile(os.path.join(cur_dir2, f))]

conv_file_dict = {}
for file in files_file:
    name, ext = os.path.splitext(file)
    items = name.split('_')
    print (items)
    if(len(items) > 2):
        conv_file = get_label(items, ext)
        if conv_file != None:
            conv_file_dict[file] =  conv_file

for key, value in conv_file_dict.items():
    shutil.copy(cur_dir2 +  key, cur_dir2 + value)
