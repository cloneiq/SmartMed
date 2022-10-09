import os

dcm_dir = 'dcms'
for root, dirs, files in os.walk(dcm_dir):
    # for dir1 in dirs:
    #     print(os.path.join(root, dir1))
    for file in files:
        print(os.path.join(root, file))
