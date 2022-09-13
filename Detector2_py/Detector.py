from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
import time

import cv2 
import numpy as numpy

class Detector:
    def __init__(self,model_type = "OD"):
        self.cfg = get_cfg()
        self.model_type = model_type
        if model_type == "OD":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        elif model_type == "IS": #instance segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        elif model_type == "KP": #instance segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        elif model_type == "LVIS": #instance segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")
        elif model_type == "PS": #instance segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cuda"

        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, imagePath):
        image = cv2.imread(imagePath)
        if self.model_type != "PS":
            predictions = self.predictor(image)
            
            viz = Visualizer(image[:,:,::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
            instance_mode = ColorMode.SEGMENTATION)
            

            output = viz.draw_instance_predictions(predictions["instances"].to("cuda"))
        else: 
            predictions, segmentInfo = self.predictor(image)["panoptic_seg"]
            viz = Visualizer(image[:,:,::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
            output = viz.draw_panoptic_seg_predictions(predictions.to("cuda"), segmentInfo)
        cv2.imshow("Results", output.get_image()[:,:,::-1])
        cv2.waitKey(0)
    def onVideo(self,videoPath):
        
        cap = cv2.VideoCapture(videoPath)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        writer1 = cv2.VideoWriter('detectron.avi', fourcc, 28, (1280, 720))
        if(cap.isOpened() == False):
            print("olmadi be")
            return
        (_, image) = cap.read()

        while _:
            start = time.time()
            if self.model_type != "PS":
                predictions = self.predictor(image)
                #print(DatasetCatalog.list())
                #print(self.cfg.DATASETS.TRAIN[0])
                viz = Visualizer(image[:,:,::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                instance_mode = ColorMode.SEGMENTATION)

                output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
            else: 

                predictions, segmentInfo = self.predictor(image)["panoptic_seg"]
                viz = Visualizer(image[:,:,::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
                #print(self.cfg.DATASETS.TRAIN[0])
                output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)
            
            end = time.time()
            totalTime = end -start
            fps = 1/totalTime
            #print("FPS", fps)
            writer1.write(output.get_image()[:,:,::-1])
            cv2.imshow("Results", output.get_image()[:,:,::-1])
            key = cv2.waitKey(1)

            if key == 27:
                cap.release()
                writer1.release()
                break
            (_,image) = cap.read()
        cap.release()
        writer1.release()