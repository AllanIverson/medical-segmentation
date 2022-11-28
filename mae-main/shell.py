import os
import multiprocessing as mp



    
def runSyn(task, force = False):
    print(task)
    os.system(task) 


# 使用传统工具配准
l = []

# for i in range(414):
#     string = './antsRegistrationSyNQuick.sh -d 3 -t a '+'-m /data2/zhanghao/oasis4_35/new_nii/image_'+str(i)+'.nii.gz '+'-f /data2/zhanghao/mae_project/test_affine/sri24.nii.gz '+'-o /data2/zhanghao/oasis4_35/affine_data/image_'+str(i)+'.nii \n'
#     l.append(string)

# with mp.Pool(20) as p:
#     p.map(runSyn , l)


# affine分割mask
for i in range(414):
    string = './antsApplyTransforms -d 3 -i /data2/zhanghao/oasis4_35/new_nii/mask4_'+str(i)+'.nii.gz -o /data2/zhanghao/oasis4_35/affine_mask4/mask4_'+str(i)+'.nii -r /data2/zhanghao/mae_project/test_affine/sri24.nii.gz -t /data2/zhanghao/oasis4_35/affine_image/image_'+str(i)+'.nii0GenericAffine.mat \n'
    l.append(string)

with mp.Pool(30) as p:
    p.map(runSyn , l)