import cv2
import math
import numpy as np
import os

limit_speed = 60
fps = 30

traffic_record_folder_name = "TrafficRecord"

if not os.path.exists(traffic_record_folder_name):
    os.makedirs(traffic_record_folder_name)
    os.makedirs(traffic_record_folder_name+"//exceeded")

speed_record_file_location = traffic_record_folder_name + "//SpeedRecord.txt"
file = open(speed_record_file_location, "w")
file.write("ID \t SPEED\n------\t-------\n")
file.close()

class Tracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0
        self.frame_count = np.zeros(1000)
        self.isCaptured = np.zeros(1000, dtype = bool)
        self.vehicle_count = 0
        self.exceeded = 0

    def track(self, objects_rect):
        objects_array = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # CHECK IF OBJECT IS DETECTED ALREADY
            same_object_detected = False

            for id, pt in self.center_points.items():
                distance = math.hypot(cx - pt[0], cy - pt[1])
                if distance < 70:
                    self.center_points[id] = (cx, cy)
                    objects_array.append([x, y, w, h, id])
                    same_object_detected = True

            # NEW OBJECT DETECTION
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_array.append([x, y, w, h, self.id_count])
                self.id_count += 1
                self.frame_count[self.id_count] = 0

        # ASSIGN NEW ID to OBJECT
        new_center_points = {}
        for item in objects_array:
            _, _, _, _, object_id = item
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()
        return objects_array

    def calcSpeed(self, id):
        if (self.frame_count[id] != 0):
            v = 14 / (self.frame_count[id] / fps)
        else:
            v = 0
        return v

    def capture(self, img, x, y, h, w, v, id):
        if (self.isCaptured[id] == False):
            self.isCaptured[id] = True
            crop_img = img[y - 5:y + h + 5, x - 5:x + w + 5]
            n = str(id) + "_speed_" + str(v)
            file_img_path = traffic_record_folder_name + '//' + n + '.jpg'
            cv2.imwrite(file_img_path, crop_img)
            self.vehicle_count += 1
            file_text = open(speed_record_file_location, "a")
                
            if (v > limit_speed):
                file_img_path2 = traffic_record_folder_name + '//exceeded//' + n + '.jpg'
                cv2.imwrite(file_img_path2, crop_img)
                file_text.write(str(id) + " \t " + str(v) + "<---exceeded\n")
                self.exceeded += 1
            else:
                file_text.write(str(id) + " \t " + str(v) + "\n")

            file_text.close()

    def getLimitSpeed(self):
        return limit_speed

    def sumary(self):
        file_text = open(speed_record_file_location, "a")
        file_text.write("\n-------------\n")
        file_text.write("-------------\n")
        file_text.write("SUMMARY\n")
        file_text.write("-------------\n")
        file_text.write("Total Vehicles :\t" + str(self.vehicle_count) + "\n")
        file_text.write("Exceeded speed limit :\t" + str(self.exceeded))
        file_text.close()
