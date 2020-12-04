import imageio
from skimage import io, filters, feature
import numpy as np
import math
from scipy.ndimage import gaussian_filter, gaussian_filter1d, sobel, maximum_filter1d,maximum_filter, gaussian_gradient_magnitude,uniform_filter
import os
from preprocessing import kitti_util
from preprocessing import generate_lidar
import cv2

DISPARITY_PATH = '../result/psmnet_disp_all/'
#DISPARITY_PATH = '../result/ganet_disparity/'
SEMANTIC_PATH = '../result/semantic_cityscapes/'
#SEMANTIC_PATH = '../result/semantic_kitti/'
SAVE_DISP_PATH = '../result/kitti_ipbasic_disp/'
SAVE_PL_PATH = '../result/psmnet_dedge_obj_pl/'
CALIB_PATH = '../KITTI/object/training/calib/'


def filter_one(input):
    # print(input.shape[0])
    output = np.zeros(input.shape)
    disp_max = np.max(input)
    depth = 1 / input
    depth_min = np.min(depth)
    depth_max = np.max(depth)
    depth_ratio = (depth - depth_min) / (depth_max - depth_min)
    mask1 = (depth_ratio < 0.25).astype(int)
    mask2 = np.logical_and(0.25 <= depth_ratio, depth_ratio < 0.75).astype(int)
    mask3 = (0.75 <= depth_ratio).astype(int)
    layer_1 = depth
    layer_2 = gaussian_filter1d(depth, sigma=10, axis=1)
    layer_3 = gaussian_filter1d(depth, sigma=10, axis=1)
    output = output + mask1 * layer_1 + mask2 * layer_2 + mask3 * layer_3
    output = 1 / output
    # mask1 = (input > disp_max*0.27).astype(int)
    # mask2 = np.logical_and( disp_max*0.1 < input,input < disp_max*0.27).astype(int)
    # #mask2 = (mask2 and (input < disp_max*0.75)).astype(int)
    # mask3 = ( input < disp_max/10).astype(int)
    # layer_1 = input
    # layer_2 = gaussian_filter1d(input, sigma = 1 ,axis = 1)
    # layer_3 = gaussian_filter1d(input, sigma= 3 ,axis = 1)
    # output = output + mask1 * layer_1 + mask2*layer_2 + mask3 * layer_3
    # output[output < 1] = 0
    # output[0, :] = 0
    # output[-1, :] = 0
    # output[:, 0] = 0
    # output[:, -1] = 0
    # for i in range(1, input.shape[0] - 1):
    #     for j in range(1, input.shape[1] - 1):
    #         if (input[i, j] != 0):
    #             block = input[i - 1:i + 2, j - 1:j + 2]
    #             # print(block)
    #             sigma =  100*(disp_max - input[i, j]) / disp_max
    #             if(sigma>0.4):
    #                 filter = gaussian_kernel(sigma)
    #                 output[i, j] = (np.sum(np.sum(filter * block)))
    #             # print(192/input[i,j])
    #             # print(block)
    #
    # print(output)

    # output = gaussian_filter(input, sigma=10)
    return output


def test_one(index):
    disp = imageio.imread(DISPARITY_PATH + index + '.png') / 256

    # disp = filter_one(disp)
    disp = test_mask(index, disp)
    #disp = semantic_mask(index, disp)
    # disp = filter_one(disp)

    print("Saved Disparity")
    imageio.imwrite('./filtered_disp/' + index + '.png', (disp * 256).astype('uint16'))
    calib_file = CALIB_PATH + index + '.txt'
    calib = kitti_util.Calibration(calib_file)
    max_high = 1
    lidar = generate_lidar.project_disp_to_points(calib, disp, max_high)
    lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
    lidar = lidar.astype(np.float32)
    lidar.tofile('{}/{}.bin'.format('./pl/', index))
    print("Saved point cloud")


def semantic_filter_all():
    file1 = open('../KITTI/object/trainval.txt', 'r')
    lines = file1.read().splitlines()
    file1.close()
    if not os.path.exists(SAVE_PL_PATH):
        os.makedirs(SAVE_PL_PATH)

    for line in lines:
        index = line
        disp = imageio.imread(DISPARITY_PATH + index + '.png') / 256
        # disp = obj_mask(index, disp)
        disp = dedge_obj_mask(index, disp)
        calib_file = CALIB_PATH + index + '.txt'
        calib = kitti_util.Calibration(calib_file)
        max_high = 1
        lidar = generate_lidar.project_disp_to_points(calib, disp, max_high)
        lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
        lidar = lidar.astype(np.float32)
        lidar.tofile(SAVE_PL_PATH + index + '.bin')
        print("Saved point cloud # " + index)

def depth_to_disp():
    DEPTH_PATH = '/home/maxc303/aer1515/ip_basic/demos/outputs/kitti_depth_ipbasic/'
    file1 = open('../KITTI/object/train.txt', 'r')
    lines = file1.read().splitlines()
    file1.close()
    if not os.path.exists(SAVE_PL_PATH):
        os.makedirs(SAVE_PL_PATH)

    for line in lines:
        index = line
        depth= imageio.imread(DEPTH_PATH + index + '.png') / 256
        # disp = obj_mask(index, disp)
        calib_file = CALIB_PATH + index + '.txt'
        calib = kitti_util.Calibration(calib_file)
        max_high = 1
        baseline = 0.54
        depth[depth==0] = -1
        disp = (calib.f_u * baseline) / depth
        disp[disp<0]=0
        disp[disp>192]=0
        imageio.imwrite(SAVE_DISP_PATH + index + '.png', (disp*256).astype('uint16'))
        print("Saved depth # " + index)




def semantic_mask(index, disp):
    s_mask = imageio.imread(SEMANTIC_PATH + 'pred_mask_' + index + '.png')
    #    c_mask = imageio.imread(SEMANTIC_PATH + 'color_mask_' + index + '.png')*256
    ground_mask = np.logical_and(s_mask >= 7, s_mask <= 10).astype(int)
    object_mask = (s_mask >= 24).astype(int)

    # #Gaussia + sobel
    # edges = sobel(gaussian_filter(s_mask,sigma=1))

    # Canny Edge
    edges = cv2.Canny(s_mask * 10, 10, 30, 10)
    edges = maximum_filter(edges, size=10)

    # Apply masks
    nonedge_mask = (edges == 0).astype(int)
    mask = np.logical_or(ground_mask, object_mask)

    #mask = object_mask
    mask = np.logical_and(nonedge_mask, mask)
    # print(np.max(edges))
    # imageio.imwrite('./filter_edges/' + index + '.png', edges.astype('uint16'))
    #    output = np.zeros(input.shape) + disp * ground_mask + disp * object_mask

    output = np.zeros(s_mask.shape) + disp * mask
    return output

def obj_mask(index, disp):
    s_mask = imageio.imread(SEMANTIC_PATH + 'pred_mask_' + index + '.png')
    ground_mask = np.logical_and(s_mask >= 7, s_mask <= 10).astype(int)
    object_mask = (s_mask >= 24).astype(int)

    # Gaussian on the object disparity
    object_disp = disp*object_mask
    # disp_ratio = (np.max(disp)-disp)/np.max(disp)
    # object_disp = maximum_filter1d(object_disp,size=10,axis=0 )
    # object_disp = gaussian_filter(object_disp,sigma =2)

    # Sobel edges
    # sobel_edges = sobel(10*object_mask,axis=0)
    # sobel_mask = sobel(sobel>0).astype(int)



    # Canny Edge
    edges = cv2.Canny(s_mask * 10, 10, 30, 10)
    edges = maximum_filter(edges, size=7)
    nonedges_mask = (edges==0).astype(int)

    # output = np.zeros(input.shape) + disp * ground_mask + disp * object_mask
    # object_disp = (disp_ratio)*object_disp + (1-disp_ratio) * disp
    mask = np.logical_and(nonedges_mask,object_mask)
    output = np.zeros(s_mask.shape) + object_disp * mask + ground_mask*disp
    # imageio.imwrite('./filter_edges/' + index + '.png', output.astype('uint16'))
    return output

def dedge_mask(index, disp):
    disp_int = np.uint8(disp)
    edges = cv2.Canny(5 * disp_int, 10, 30, 10)
    edges = maximum_filter(edges, size=3)
    nonedges_mask = (edges==0).astype(int)
    output = np.zeros(disp.shape)+ disp*nonedges_mask
    return output

def dedge_obj_mask(index, disp):
    s_mask = imageio.imread(SEMANTIC_PATH + 'pred_mask_' + index + '.png')
    ground_mask = np.logical_and(s_mask >= 7, s_mask <= 10).astype(int)
    object_mask = (s_mask >= 24).astype(int)

    disp_int = np.uint8(disp)
    edges = cv2.Canny(5 * disp_int, 10, 30, 10)
    edges = maximum_filter(edges, size=3)
    nonedges_mask = (edges==0).astype(int)

    mask = np.logical_and(nonedges_mask,object_mask)
    output = np.zeros(disp.shape)+ disp* mask  + ground_mask * disp
    return output


def test_mask(index, disp):
    s_mask = imageio.imread(SEMANTIC_PATH + 'pred_mask_' + index + '.png')
    ground_mask = np.logical_and(s_mask >= 7, s_mask <= 10).astype(int)
    object_mask = (s_mask >= 24).astype(int)

    # Gaussian on the object disparity
    object_disp = disp*object_mask
    # disp_ratio = (np.max(disp)-disp)/np.max(disp)
    # object_disp = maximum_filter1d(object_disp,size=10,axis=0 )
    # object_disp = gaussian_filter(object_disp,sigma =2)

    # Sobel edges
    # sobel_edges = sobel(10*object_mask,axis=0)
    # sobel_mask = sobel(sobel>0).astype(int)



    # Canny Edge
    # edges = cv2.Canny(s_mask * 10, 10, 30, 10)
    disp_int =  np.uint8(disp)
    edges = cv2.Canny(5*disp_int, 10, 30, 10)
    edges = maximum_filter(edges, size=3)

    nonedges_mask = (edges==0).astype(int)

    #imageio.imwrite('./filter_edges/' + index + '.png', (256*disp_ratio).astype('uint16'))
    imageio.imwrite('./filter_edges/' + index + '.png', edges.astype('uint16'))

       # output = np.zeros(input.shape) + disp * ground_mask + disp * object_mask
    # object_disp = (disp_ratio)*object_disp + (1-disp_ratio) * disp
    mask = np.logical_and(nonedges_mask,object_mask)
    output = np.zeros(s_mask.shape) + object_disp * mask + ground_mask * disp
    output = disp*nonedges_mask
    # imageio.imwrite('./filter_edges/' + index + '.png', output.astype('uint16'))

    return output
def apply_filter():
    file1 = open('../tools/train.txt', 'r')
    lines = file1.read().splitlines()
    file1.close()

    for line in lines:
        a = 0


def gaussian_kernel(sigma):
    mag = 1 / (2 * math.pi * sigma * sigma)
    g1 = mag * math.exp(-1 / (sigma * sigma))
    g2 = mag * math.exp(-1 / (2 * sigma * sigma))
    g3 = mag * math.exp(0)
    # print(g1,g2,g3)
    filter = np.zeros([3, 3])
    filter[0, 0] = g1
    filter[0, 2] = g1
    filter[2, 0] = g1
    filter[2, 2] = g1
    filter[1, 1] = g3
    filter[0, 1] = g2
    filter[1, 0] = g2
    filter[2, 1] = g2
    filter[1, 2] = g2
    filter_sum = sum(sum(filter))

    filter = (1 / filter_sum) * filter

    return filter


if __name__ == '__main__':
    semantic_filter_all()
    #test_one('000006')
   # depth_to_disp()
#    semantic_mask()
# print(gaussian_kernel(0.99))
