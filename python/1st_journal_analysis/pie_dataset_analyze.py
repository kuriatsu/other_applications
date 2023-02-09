
#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import pickle
import numpy
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math

def getXmlRoot(filename):
    tree = ET.parse(filename)
    return tree.getroot()

####################################
# average speed analyze
####################################
annt_vehicle_1 = getXmlRoot("/home/kuriatsu/Documents/experiment/pie/annotations_vehicle/set04/video_0001_obd.xml")
annt_vehicle_2 = getXmlRoot("/home/kuriatsu/Documents/experiment/pie/annotations_vehicle/set04/video_0002_obd.xml")
buf = []
acc = []
for frame in annt_vehicle_1:
    if float(frame.get("GPS_speed")) > 3.6:
        acc.append([float(frame.get("accX")), float(frame.get("accY")), float(frame.get("accZ"))])
        buf.append(float(frame.get("GPS_speed")))
time = len(annt_vehicle_1)/30
max_vel = max(buf)
min_vel = min(buf)
ave_vel = ave_vel = sum(buf)/len(buf)
std_vel = np.std(buf)
print(f"time:{time},max:{max_vel}, min:{min_vel}, ave:{ave_vel}, std:{std_vel}")

buf = []
acc = []
for frame in annt_vehicle_1:
    acc.append((float(frame.get("accX"))**2 + float(frame.get("accY"))**2)**0.5 / 9.8)
    # acc.append([float(frame.get("accX")), float(frame.get("accY")), float(frame.get("accZ"))])
    buf.append(float(frame.get("GPS_speed")))
sns.lineplot(x=np.arange(len(buf)), y=buf)
sns.lineplot(x=np.arange(len(acc[:])), y=[row[0] for row in acc[:]])
sns.lineplot(x=np.arange(len(acc[:])), y=[row[1] for row in acc[:]])
sns.lineplot(x=np.arange(len(acc[:])), y=[row[2] for row in acc[:]])
sns.lineplot(x=np.arange(len(acc)), y=acc)
max(acc)
sum(acc)/len(acc)

buf = []
for frame in annt_vehicle_2:
    if float(frame.get("GPS_speed")) > 3.6:
        buf.append(float(frame.get("GPS_speed")))
time = len(annt_vehicle_2)/30
max_vel = max(buf)
min_vel = min(buf)
ave_vel = ave_vel = sum(buf)/len(buf)
std_vel = np.std(buf)
print(f"time:{time},max:{max_vel}, min:{min_vel}, ave:{ave_vel}, std:{std_vel}")


####################################
# present time analyze
####################################
annt_attrib_1 = getXmlRoot("/home/kuriatsu/Documents/experiment/pie/annotations_attributes/set04/video_0001_attributes.xml")
annt_attrib_2 = getXmlRoot("/home/kuriatsu/Documents/experiment/pie/annotations_attributes/set04/video_0002_attributes.xml")
buf = []
for pedestrian in annt_attrib_1:
    buf.append((float(pedestrian.get("critical_point")) - float(pedestrian.get("exp_start_point")))/30.0)
print(sum(buf)/len(buf), np.std(buf), max(buf), min(buf))

buf = []
for pedestrian in annt_attrib_2:
    buf.append((float(pedestrian.get("critical_point")) - float(pedestrian.get("exp_start_point")))/30.0)
print(sum(buf)/len(buf), np.std(buf), max(buf), min(buf))
