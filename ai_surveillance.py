import cv2
import os
from ultralytics import YOLO
import bbox_visualizer as bbv
import numpy as np
from typing import Union
from collections import Counter
import time
#https://www.youtube.com/live/zzJjopSjIMc?feature=share
# url = 'https://www.youtube.com/watch?v=zzJjopSjIMc'  # Replace with your desired YouTube video URL

# video = pafy.new(url)
# best = video.getbest(preftype='mp4')  # Get the best quality video


random_colors = {i:tuple(map(int,col)) for i , col in enumerate(np.random.randint(3,255,(100,3)))}




class ObjectTracker():
    def __init__(self,video_path:os.path,model:os.path) -> None:
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_video = cv2.VideoWriter('output_video.mp4', self.fourcc, self.fps, (self.frame_width, self.frame_height))
        self.model = YOLO(model)
        self.position = {}
        self.random_color = np.random.randint(0,233,(100,3))
        self.zero_counter = []
        self.map = {0:'person'}
        self.counter = {'person':0}
        self.lane_crop_region = [(0,0),(0,self.frame_height),(self.frame_width,self.frame_height),(self.frame_width,0)]
        self.car_ids = set()
        self.truck_id = set()
        self.bike_id = set()
        self.bus_id = set()
    
    def set_specific_object_color(self,object_class_no):
        check_list = []
        if object_class_no not in  check_list:
            check_list.append(object_class_no)
            return tuple(map(int,self.random_color[object_class_no]))
        else:
            return tuple(map(int,self.random_color[object_class_no]))
        
    def track(self,draw_tail:bool=False,confidence=0.50,iou=.50,skip_frames:Union[None,int]=None,slow_down_video=1,add_label=False):
        if skip_frames:
            count = 1
            
        while True:
            start_time = time.time()
            ret, frame = self.cap.read()
            
            out_x , out_y = self.frame_width//2,self.frame_height//2
            cv2.polylines(frame, [np.array(self.lane_crop_region)], isClosed=True, color=(0, 255, 0), thickness=1,lineType=cv2.LINE_AA)
            cv2.polylines(frame, [np.array([[0,self.frame_height//2],[self.frame_width,self.frame_height//2]])], isClosed=False, color=(0, 255, 0),thickness=1,lineType=cv2.LINE_AA)
            cv2.putText(frame,'Gate',(out_x,out_y+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            
            if not ret:
                break
            
            if skip_frames:
                count += 1
                if count % skip_frames != 0:
                    continue
            
            results = self.model.track(frame, persist=True,conf=confidence,iou=iou,tracker='bytetrack.yaml')
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clist= results[0].boxes.cls.cpu().numpy().astype(int)
           
       
            for box, id , classes in zip(boxes, ids , clist):
       
                if classes == 0:
              
                    bboxs = (box[0],box[1],box[2],box[3])
                    middle_x , middle_y = int((box[0]+box[2])/2) , int((box[1]+box[3])/2)
                    cv2.circle(frame,(middle_x,middle_y),2,(0,0,255),2)
                    center = (middle_x,middle_y)
                    cv2.rectangle(frame , (0,0) , (280,120),(0,155,0),-1)
                    
                    # if classes == 2:
                        
                    color = (255,0,0)        
                    self.truck_id = set()
                    self.bike_id = set()
                    self.bus_id = set()
                    result = cv2.pointPolygonTest(np.array(self.lane_crop_region), (middle_x,middle_y), measureDist=False)
                        
                    if result > 0:
                            # self.car_counter += 1
                            self.car_ids.add(id)
                            
                            cv2.rectangle(frame, (bboxs[0], bboxs[1]), (bboxs[2], bboxs[3]),color=color,thickness=2,lineType=cv2.LINE_AA)
                            #bbv.add_label(frame,f'PersonID:{str(id)}', bbox=bboxs, top=False,text_color=random_colors,draw_bg=True,)
                            
                    
                    # if classes == 7:
                        
                    #     color = (255,255,0)
                    #     results = cv2.pointPolygonTest(np.array(self.lane_crop_region), (middle_x,middle_y), measureDist=False)
                        
                    #     if results > 0:
                    #         # self.car_counter += 1
                    #         self.truck_id.add(id)
                    #         cv2.rectangle(frame, (bboxs[0], bboxs[1]), (bboxs[2], bboxs[3]),color=color,thickness=2,lineType=cv2.LINE_AA)
                    #         bbv.add_label(frame,f'TruckID:{str(id)}', bbox=bboxs, top=False,text_color=random_colors,draw_bg=True,)
                            
                    # if classes == 3:
                        
                    #     color = (255,0,255)
                    #     results = cv2.pointPolygonTest(np.array(self.lane_crop_region), (middle_x,middle_y), measureDist=False)
                        
                    #     if results > 0:
                    #         # self.car_counter += 1
                    #         self.bike_id.add(id)
                    #         cv2.rectangle(frame, (bboxs[0], bboxs[1]), (bboxs[2], bboxs[3]),color=color,thickness=2,lineType=cv2.LINE_AA)
                    #         bbv.add_label(frame,f'BikeID:{str(id)}', bbox=bboxs, top=False,text_color=random_colors,draw_bg=True,)
                    
                    # if classes == 5:
                        
                    #     color = (0,255,255)
                    #     results = cv2.pointPolygonTest(np.array(self.lane_crop_region), (middle_x,middle_y), measureDist=False)
                        
                    #     if results > 0:
                    #         # self.car_counter += 1
                    #         self.bus_id.add(id)
                    #         cv2.rectangle(frame, (bboxs[0], bboxs[1]), (bboxs[2], bboxs[3]),color=color,thickness=2,lineType=cv2.LINE_AA)
                    #         bbv.add_label(frame,f'BusID:{str(id)}', bbox=bboxs, top=False,text_color=random_colors,draw_bg=True,)
                            
                    random_colors = self.set_specific_object_color(classes)
                    
                    # cv2.putText(frame,f'{self.map[0]} : {len(person)}',(20,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                    # cv2.putText(frame,f'{self.map[2]} : {len(car)}',(20,120),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
       
                    # cv2.rectangle(frame, (bboxs[0], bboxs[1]), (bboxs[2], bboxs[3]),color=random_colors,thickness=2,lineType=cv2.LINE_AA)
                 
                    if add_label:
                        bbv.add_label(frame,f'Person id:{str(id)}', bbox=bboxs, top=False,text_color=random_colors,draw_bg=True,)
            
                    if draw_tail:
                        if id not in self.position:
                            self.position[id] = [center]
                        if id in self.position:
                            self.position[id].append(center)
                            #print(self.position)
                        for ids , pos in self.position.items():
                            for i, p in enumerate(pos): 
                                
                                cv2.circle(frame,p,2,random_colors,2)
                     
                        
            #return self.position    
            cv2.putText(frame,f'Person: {str(len(self.car_ids))}',(20,70),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)       
            # cv2.putText(frame,f'truck : {str(len(self.truck_id))}',(20,110),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)       
            # cv2.putText(frame,f'bike : {str(len(self.bike_id))}',(20,150),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)       
            # cv2.putText(frame,f'bus : {str(len(self.bus_id))}',(20,190),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)       
            
            FPS = 1.0 / (time.time() - start_time)
            cv2.putText(frame,f'FPS: {round(FPS)}',(20,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            self.output_video.write(frame)
            cv2.imshow("frame", frame)
            if cv2.waitKey(slow_down_video) & 0xFF == ord("q"):
                break
        self.cap.release()
        self.output_video.release()
        #return self.position

if __name__ == '__main__':
    tracker = ObjectTracker(video_path='people.mp4',model='yolov8l.pt') # models name ['yolov8l.pt','yolov8m.pt','yolov8s.pt']
    tracker.track(draw_tail=False,confidence=0.5,iou=0.3,skip_frames=None,slow_down_video=1,add_label=True)






