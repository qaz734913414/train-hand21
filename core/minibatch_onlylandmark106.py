import cv2
import threading
from tools import image_processing
import numpy as np
import numpy.random as npr
import math
import os,sys
sys.path.append(os.getcwd())
from config import config
import tools.image_processing as image_processing

class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.ims, self.landmarks, self.landmarks_vis = self.func(*self.args)
    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.ims, self.landmarks, self.landmarks_vis
        except Exception:
            return None

def get_minibatch_thread(imdb, im_size):
    num_images = len(imdb)
    processed_ims = list()
    landmark_reg_target = list()
    landmark_vis = list()
    #print(num_images)
    for i in range(num_images):
        im,landmark,vis = augment_for_one_image(imdb[i],im_size)
        im_tensor = image_processing.transform(im,True)
        processed_ims.append(im_tensor)
        landmark_reg_target.append(landmark)
        landmark_vis.append(vis)

    return processed_ims, landmark_reg_target, landmark_vis

def get_minibatch(imdb, im_size, thread_num = 4):
    num_images = len(imdb)
    thread_num = max(1,thread_num)
    num_per_thread = math.ceil(float(num_images)/thread_num)
    #print(num_per_thread)
    threads = []
    for t in range(thread_num):
        start_idx = int(num_per_thread*t)
        end_idx = int(min(num_per_thread*(t+1),num_images))
        cur_imdb = [imdb[i] for i in range(start_idx, end_idx)]
        cur_thread = MyThread(get_minibatch_thread,(cur_imdb,im_size))
        threads.append(cur_thread)
    for t in range(thread_num):
        threads[t].start()

    processed_ims = list()
    landmark_reg_target = list()
    landmark_vis = list()

    for t in range(thread_num):
        cur_process_ims, cur_landmark_reg_target, cur_landmark_vis = threads[t].get_result()
        processed_ims = processed_ims + cur_process_ims
        landmark_reg_target = landmark_reg_target + cur_landmark_reg_target    
        landmark_vis = landmark_vis + cur_landmark_vis
    
    im_array = np.vstack(processed_ims)
    landmark_target_array = np.vstack(landmark_reg_target)
    landmark_vis_array = np.vstack(landmark_vis)
    
    data = {'data': im_array}
    label = {}
    label['landmark_target'] = landmark_target_array
    label['landmark_vis'] = landmark_vis_array

    return data, label

def augment_for_one_image(annotation_line, size):
    annotation = annotation_line.strip().split()
    img_path = config.root+'/data/%s/'%config.landmark_img_set+annotation[0]
    #print img_path
    img = cv2.imread(img_path)
    width = img.shape[1]
    height = img.shape[0]
    landmark = np.array(annotation[1:64],dtype=np.float32)
    landmark_x = landmark[0::3]
    landmark_y = landmark[1::3]
    vis_x = landmark[2::3]
    max_x = max(landmark_x)
    min_x = min(landmark_x)
    max_y = max(landmark_y)
    min_y = min(landmark_y)
    cx = 0.5*(max_x+min_x)
    cy = 0.5*(max_y+min_y)
    w = max_x-min_x
    h = max_y-min_y
    bbox_size = max(h,w)
    x1 = int(cx - bbox_size*0.5)
    y1 = int(cy - bbox_size*0.5)
    w = bbox_size
    h = bbox_size
 

    init_rot = 0
    
    cur_angle = npr.randint(int(config.min_rot_angle - init_rot),int(config.max_rot_angle - init_rot)+1)
    try_num = 0
    cur_sample_num = 0
    base_num = 1
    force_accept = 0
    while cur_sample_num < base_num:
        try_num += 1
        if try_num > base_num*1000:
            force_accept = 1
            break
        rot_landmark_x,rot_landmark_y = image_processing.rotateLandmark21(cx,cy,landmark_x,landmark_y, cur_angle,1)
        rot_max_x = max(rot_landmark_x)
        rot_min_x = min(rot_landmark_x)
        rot_max_y = max(rot_landmark_y)
        rot_min_y = min(rot_landmark_y)
        rot_cx = 0.5*(rot_max_x+rot_min_x)
        rot_cy = 0.5*(rot_max_y+rot_min_y)
        rot_w = rot_max_x-rot_min_x
        rot_h = rot_max_y-rot_min_y
        rot_bbox_size = max(rot_h,rot_w)
        rot_x1 = int(rot_cx - rot_bbox_size*0.5)
        rot_y1 = int(rot_cy - rot_bbox_size*0.5)
        rot_w = rot_bbox_size
        rot_h = rot_bbox_size
        cur_size = int(npr.randint(15, 21)*0.1*rot_bbox_size)
        up_border_size = int(-cur_size*0.15)
        down_border_size = int(-cur_size*0.15)
        left_border_size = int(-cur_size*0.15)
        right_border_size = int(-cur_size*0.15)
        #up_border_size = int(cur_size*0.05)
        #down_border_size = int(cur_size*0.05)
        #left_border_size = int(cur_size*0.05)
        #right_border_size = int(cur_size*0.05)

        # delta here is the offset of box center
        #delta_x = npr.randint(-int(rot_w * 0.35), int(rot_w * 0.35)+1)
        #delta_y = npr.randint(-int(rot_h * 0.35), int(rot_h * 0.35)+1)
        delta_x = npr.randint(-int(rot_w * 0.20), int(rot_w * 0.20)+1)
        delta_y = npr.randint(-int(rot_h * 0.20), int(rot_h * 0.20)+1)
        #delta_x = npr.randint(-int(rot_w * 0.02), int(rot_w * 0.02)+1)
        #delta_y = npr.randint(-int(rot_h * 0.02), int(rot_h * 0.02)+1)
		
		
        nx1 = int(max(rot_x1 + rot_w / 2 + delta_x - cur_size / 2, 0))
        ny1 = int(max(rot_y1 + rot_h / 2 + delta_y - cur_size / 2, 0))
        nx2 = nx1 + cur_size
        ny2 = ny1 + cur_size

        if nx2 > width or ny2 > height:
            continue
        ignore = 0
        max_rot_landmark_x = max(rot_landmark_x)
        min_rot_landmark_x = min(rot_landmark_x)
        max_rot_landmark_y = max(rot_landmark_y)
        min_rot_landmark_y = min(rot_landmark_y)
        if min_rot_landmark_x < nx1+left_border_size or max_rot_landmark_x >= nx1 + cur_size-right_border_size:
            ignore = 1
        if min_rot_landmark_y < ny1+up_border_size or max_rot_landmark_y >= ny1 + cur_size-down_border_size:
            ignore = 1
												
        if ignore == 1:
            continue
        landmark_x_dis = max_rot_landmark_x - min_rot_landmark_x
        landmark_y_dis = max_rot_landmark_y - min_rot_landmark_y
        tmp_dis = landmark_x_dis*landmark_x_dis + landmark_y_dis*landmark_y_dis
        #if tmp_dis < 0.64*cur_size*cur_size:
        if tmp_dis < 1.00*cur_size*cur_size:
            continue
			
        offset_x = (rot_landmark_x - nx1+0.5)/float(cur_size)
        offset_y = (rot_landmark_y - ny1+0.5)/float(cur_size)
        
        rot_img,_,_ = image_processing.rotateWithLandmark21(img,cx,cy,landmark_x,landmark_y, cur_angle,1)
        cropped_im = rot_img[ny1 : ny2, nx1 : nx2, :]
        resized_im = cv2.resize(cropped_im, (size, size), interpolation=cv2.INTER_LINEAR)
        
        cur_sample_num += 1

    if force_accept == 1:
        ny1 = max(0,y1)
        ny2 = int(min(height,y1+h))
        nx1 = max(0,x1)
        nx2 = int(min(width,x1+w))
        w = nx2-nx1
        h = ny2-ny1
        #print ny1,ny2,nx1,nx2
        cropped_im = img[ny1 : ny2, nx1 : nx2, :]
        resized_im = cv2.resize(cropped_im, (size, size), interpolation=cv2.INTER_LINEAR)
        offset_x = (landmark_x - nx1+0.5)/float(cur_size)
        offset_y = (landmark_y - ny1+0.5)/float(cur_size)
    
    
    landmark[0::2] = offset_x
    landmark[1::2] = offset_y
	
    vis = landmark.copy()
    vis[0::2] = vis_x
    vis[1::2] = vis_x
		
    
    if config.enable_blur:
        #kernel_size = npr.randint(-5,4)*2+1
        kernel_size = npr.randint(-5,13)*2+1
        if kernel_size >= 3:
            blur_im = cv2.GaussianBlur(resized_im,(kernel_size,kernel_size),0)
            resized_im = blur_im
    
    if config.enable_black_border:
        black_size = npr.randint(0,int(size*0.5))
        if npr.randint(0,2) == 0:
            resized_im[:,0:black_size,:] = 128
        else:
            resized_im[:,(size-black_size):size,:] = 128
    return resized_im,landmark,vis