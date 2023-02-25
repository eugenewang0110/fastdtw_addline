import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections
from scipy import stats
import pandas as pd
import statsmodels.stats.anova as anova
from PIL import Image
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


proposed_path = './output/CoG_only2/CNN/' 
GT_path = './output/pred_GT/'


DTWs = []
path = []


def get_pts(i, GT_or_NOT, path):
    points = []
    if GT_or_NOT == True:
        img = np.array(Image.open(path + 'img' + str(i) + '.jpg'))
    else:
        img = np.array(Image.open(path + 'pred_img' + str(i) + '.jpg'))
    img = (img > 128) * 255
    raw_pts = np.array(np.where(img==255))   
    for j in range(len(raw_pts[1])):
        points.append([raw_pts[1][j], raw_pts[0][j]])
    points = np.array(points)
    return points



def get_array(path1, path2):
    for i in range(int(len(os.listdir(path2)))):
        pts1 = get_pts(i+1, False, path1)
        pts2 = get_pts(i+1, True, path2)
        distance, path = fastdtw(pts1, pts2, dist=euclidean)
        print("-----------------------------")  
        print(f"distance + {distance}")
        print(f"average DTW = {distance/len(path)}")
        
        
        ### visualization ###
        plt.figure()
        plt.scatter(pts2[:, 0], pts2[:, 1], marker='o', s=10, color='r')
        plt.scatter(pts1[:, 0], pts1[:, 1], marker='o', s=10, color='b')
        ax = plt.gca()
        for j in range(len(path)):
            path_num =  path[j]
            position_line1 = path_num[0]
            position_line2 = path_num[1]
            cor_line1 = pts1[position_line1]
            cor_line2 = pts2[position_line2]
            plt.plot([cor_line1.item(0), cor_line2.item(0)], [cor_line1.item(1), cor_line2.item(1)],'g')
        ax.set_xlim([0, 64])
        ax.set_ylim([64, 0])
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_xlabel('x', labelpad=5, rotation=0)
        ax.set_ylabel('y', labelpad=10, rotation=360)
        ax.set_aspect('equal', adjustable='box')
        plt.savefig('./output/visualize_DTW/' + str(i+1) + '.jpg', dpi=300) #CNN
        print("-----------------------------")

    return DTWs, path

if __name__ == "__main__":
    ##### Get DTW #####
    proposed_DTWs1 = get_array(proposed_path, GT_path)

    


'''
    ##### 分散分析 #####
    subjects = ["img1","img2","img3","img4","img5","img6","img7","img8","img9","img10"]
    points = np.array(CoG_DTWs + proposed_DTWs1) # 4or5
    conditions = np.repeat(["CoG","proposed1"], len(CoG_DTWs)) # 4or5
    subjects = np.array(subjects + subjects) # 4or5
    # print(subjects)
    # print(conditions)
    # print(points)

    df = pd.DataFrame({"Points":points, "Conditions":conditions, "Subjects":subjects})
    aov=anova.AnovaRM(df, 'Points', 'Subjects', ['Conditions'])
    result=aov.fit()

    print(result)
    
    ##### DTWの表示および多重比較 #####
    # print('DTW_ave: ', np.average(np.array(proposed_DTWs1)), 'DTW_std: ', np.std(np.array(proposed_DTWs1)))
    print(stats.ttest_rel(np.array(CoG_DTWs), np.array(proposed_DTWs1)))

'''