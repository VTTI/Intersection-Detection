# -*- coding: utf-8 -*-
"""
Date:3/24/2022 
Author: Shreyas Bhat
E-mail:sbhat@vtti.vt.edu
Description:    
    ######################################################
    ## Class FrameDataExtractor                         ##
    ## Read Annotattion files to sample and extract     ##
    ## Intersection frames and non intersection frames  ##
    ###################################################### 
"""

import pandas as pd 
import os 
import argparse as ap
import cv2
import glob 
import pathlib
import random

## Function to parse input arguments ##
def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("--path_to_videos","-pv", help = "Path to videos" , default = "video_data")
    parser.add_argument("--path_to_annotation","-pa", help = "Path to annotation csv file", default = "video_data/Annotation/intersection_video_LG.csv")
    parser.add_argument("--save_directory","-sd", help = "Path to save directory", default = "data")
    parser.add_argument("--num_samples","-ns", help = "Number of frames to sample", default = "all")
    parser.add_argument("--data_name","-dn", help = "Unique name for data being extracted", default = "Front_Video")
    
    args = parser.parse_args()
    
    return args.path_to_videos, args.path_to_annotation, args.save_directory , args.num_samples ,args.data_name

class FrameDataExtractor:
    
    def __init__(self,videos_path,annotation_path,save_directory,data_name):
        self.videos_path = videos_path
        self.annotation_path = annotation_path
        self.save_directory = save_directory
        self.data_name = data_name
    
    def saveLog(self, msg):
        log_path = pathlib.Path("RunLog")
        log_path.mkdir(parents=True,exist_ok=True)
        log_path = os.path.join(log_path.as_posix(), "RunLog2.0.txt")
        
        if os.path.isfile(log_path):
            logfile = open(log_path,'a')
            logfile.write(str(msg)+'\n')
        else:
            logfile = open(log_path,'w')
            logfile.write(str(msg)+'\n')
        logfile.close()
    
    ## Method to create necessary directories and write images
    def saveFrame(self,img,filename,tod,frame_count, intersection):
        
        if tod == 'D':
            tod = 'Day'
        elif tod == 'N':
            tod = 'Night'

        if intersection:
            directory = "Intersection"
        else:
            directory = "Non-Intersection"
        
        self.saveLog("{}_Frame:{}".format(directory,frame_count))
        save_path =  pathlib.Path(os.path.join(self.save_directory,directory, self.data_name, tod , self.data_name+'_Images'))
        save_path.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(save_path.as_posix()+'/{}'.format(filename.split('.')[0]) + '_f{}.jpg'.format(frame_count), img)
        
    ## Method to get num_samples of intersection and non-intersection frames from video 
    def getFrames(self, video,clips,n_samples,intersection_only = False):
        ## Setup
        cap = cv2.VideoCapture(video)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        ## time of day to save in the appropraite directory
        tod = clips[0][-1]
        self.saveLog("{}_Video:{}".format(tod,video))
        
        filename = os.path.basename(video)
        ## Get all frames between entry and exit points for each intersection clip
        int_frames = []
        for key in clips.keys():
            int_times = clips[key]
            ## Ignore exit second, evaluate frames from start to end-1 seconds
            for frame in range(int(int_times[0] * frame_rate),int(int_times[1]*frame_rate)) :
                int_frames.append(frame)
        
        ## Make alist of non intersection frames in video
        non_int_frames = []
        for frame in range (total_frame_count):
            if frame not in int_frames:
                non_int_frames.append(frame)
        
        ## Get number of sampless to extract 
        if n_samples == 1000 : ## 1000 indicates all samples
            num_samples = len(int_frames)
        else:
            num_samples = len(clips)*n_samples

        ## Randomly sample num samples of frames form int and non-int
        int_out_frames = []
        non_int_out_frames = []
        if len(int_frames) >= num_samples :
            int_out_frames = random.sample(int_frames,num_samples)
        else:
            int_out_frames = random.sample(int_frames,len(int_frames))

        if len(non_int_frames) >= num_samples :
            non_int_out_frames = random.sample(non_int_frames,num_samples)
        else:
            int_out_frames = random.sample(non_int_frames,len(non_int_frames))
        
        frame_count = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret==True:
                ## Saves only intersection frames
                if intersection_only:
                    ## Intersection 
                    ## Conitions on randomly sample num_sample frames
                    if frame_count in int_out_frames  :
                        self.saveFrame(frame, filename, tod, frame_count, intersection=True)

                ## Saves both intersection and non_intersection clips
                ## extracts equal number of both
                else:
                    ## Intersection
                    ## Conitions on randomly sample num_sample frames
                    if frame_count in int_out_frames :
                        self.saveFrame(frame, filename, tod, frame_count, intersection=True)
                                
                    ## Non- Intersection
                    elif frame_count in non_int_out_frames  :
                        self.saveFrame(frame, filename, tod, frame_count, intersection=False)
                            
                            
                frame_count += 1
            else:
                break

        cap.release()
        
    ## main method that puts evrything together    
    def main(self,num_samples):
        ## videos
        videos_path = os.path.join(self.videos_path,"**","*.mp4")
        videos = glob.iglob(videos_path,recursive=True)    
        ## annotations
        annotation_df = pd.read_csv(self.annotation_path)
        
        for video in videos:
            vid_name = os.path.basename(video)
            annotation = annotation_df.loc[annotation_df["Name"] == vid_name]
            time_string = annotation["Events"].iloc[0]
            tod = str(annotation["Day or Night (D/N)"].iloc[0])
            time_entries = [int(char) for char in time_string.split(',') if char.isdigit()]
            int_clip_times = {}
            
            ## Check for at least one pair of start,end time
            if len(time_entries) > 1 :
                key = 0
                for idx in range(0,len(time_entries),2):
                    start = time_entries[idx]
                    end = time_entries[idx+1] 
                    int_clip_times[key] = (start,end,tod)
                    key+=1
            
            self.getFrames(video,int_clip_times,num_samples,False)
        
if __name__ == "__main__":
    
    path , an_path , save_path , num_samples , data_name = parse_args()
    if num_samples == 'all':
        num_samples = 1000 ## makring with flag 1000
    else:
        num_samples = int(num_samples)
    FDE = FrameDataExtractor(path, an_path, save_path, data_name)
    FDE.saveLog("Parameters used for data extraction:\nVideos_path:{}\nAnnotation_path:{}\nData_save_path:{}\nNum_samples:{}\nDataset_name:{}".format(path,an_path,save_path,num_samples,data_name))
    try:
        FDE.main(num_samples)
    except Exception as ex:
        FDE.saveLog(str(ex))
