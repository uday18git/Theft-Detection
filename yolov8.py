# import torch
# import numpy as np
# import cv2
# from time import time
# from ultralytics import YOLO

# import supervision as sv
# from supervision.draw.color import ColorPalette
# # from supervision.draw.box import BoxAnnotator
# # from supervision import Detections, BoxAnnotator


# class ObjectDetection:

#     def __init__(self, capture_index):
       
#         self.capture_index = capture_index
        
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         print("Using Device: ", self.device)
        
#         self.model = self.load_model()
        
#         self.CLASS_NAMES_DICT = self.model.model.names
    
#         self.box_annotator = sv.BoxAnnotator(color=ColorPalette(), thickness=3, text_thickness=3, text_scale=1.5)
    

#     def load_model(self):
       
#         model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n model
#         model.fuse()
    
#         return model


#     def predict(self, frame):
       
#         results = self.model(frame)
        
#         return results
    

#     def plot_bboxes(self, results, frame):
        
#         xyxys = []
#         confidences = []
#         class_ids = []
        
#         # Extract detections for person class
#         for result in results[0]:
#             class_id = result.boxes.cls.cpu().numpy().astype(int)
            
#             if class_id == 0:
                
#                 xyxys.append(result.boxes.xyxy.cpu().numpy())
#                 confidences.append(result.boxes.conf.cpu().numpy())
#                 class_ids.append(result.boxes.cls.cpu().numpy().astype(int))
            
        
#         # Setup detections for visualization
#         detections = sv.Detections(
#                     xyxy=results[0].boxes.xyxy.cpu().numpy(),
#                     confidence=results[0].boxes.conf.cpu().numpy(),
#                     class_id=results[0].boxes.cls.cpu().numpy().astype(int),
#                     )
        
    
#         # Format custom labels
#         self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
#         for _, confidence, class_id, tracker_id
#         in detections]
        
#         # Annotate and display frame
#         frame = self.box_annotator.annotate(frame=frame, detections=detections, labels=self.labels)
        
#         return frame
    
    
    
#     def __call__(self):

#         cap = cv2.VideoCapture(self.capture_index)
#         assert cap.isOpened()
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
      
#         while True:
          
#             start_time = time()
            
#             ret, frame = cap.read()
#             assert ret
            
#             results = self.predict(frame)
#             frame = self.plot_bboxes(results, frame)
            
#             end_time = time()
#             fps = 1/np.round(end_time - start_time, 2)
             
#             cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
#             cv2.imshow('YOLOv8 Detection', frame)
 
#             if cv2.waitKey(5) & 0xFF == 27:
                
#                 break
        
#         cap.release()
#         cv2.destroyAllWindows()
        
        
    
# detector = ObjectDetection(capture_index=0)
# detector()
import cv2
import torch
import numpy as np
from time import time
from ultralytics import YOLO


class ObjectDetection:

    def __init__(self):
       
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.model = self.load_model()
        
        self.CLASS_NAMES_DICT = self.model.model.names

        # Define a list of colors
        self.colors = [
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 0, 255),  # Red
            (255, 255, 0),  # Cyan
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
        ]
    
        self.thickness = 3
        self.text_thickness = 3
        self.text_scale = 1.5
    

    def load_model(self):
       
        model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n model
        model.fuse()
    
        return model


    def predict(self, frame):
       
        results = self.model(frame)
        
        return results
    

    def plot_bboxes(self, results, frame):
        
        xyxys = []
        confidences = []
        class_ids = []
            
        # Extract detections for person class
        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)
                
            if class_id == 0:
                    
                xyxys.append(result.boxes.xyxy.cpu().numpy())
                confidences.append(result.boxes.conf.cpu().numpy())
                class_ids.append(result.boxes.cls.cpu().numpy().astype(int))
                
            
        # Draw bounding boxes around the detected objects
        for i, xyxy in enumerate(xyxys):
            color = self.colors[i % len(self.colors)]
            xyxy = tuple(map(int, xyxy))
            cv2.rectangle(frame, xyxy[0:2], xyxy[2:4], color, self.thickness)
            cv2.putText(frame, f"{self.CLASS_NAMES_DICT[class_ids[i]]} {confidences[i]:0.2f}", (int(xyxy[0]), int(xyxy[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, color, self.text_thickness)
            
        return frame

    
    
    
    def __call__(self):

        cap = cv2.VideoCapture(0)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
      
        while True:
          
            start_time = time()
            
            ret, frame = cap.read()
            assert ret
            
            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame)
            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
             
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
            cv2.imshow('YOLOv8 Detection', frame)
 
            if cv2.waitKey(5) & 0xFF == 27:
                
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        
    
detector = ObjectDetection()
detector()
