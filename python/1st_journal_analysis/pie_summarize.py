#! /usr/bin/python3
# -*- coding: utf-8 -*-

import pickle
import pandas as pd
import glob
import os
import math
import xml.etree.ElementTree as ET


def getXmlRoot(filename):

    # try:
    tree = ET.parse(filename)
    return tree.getroot()


extracted_data = pd.DataFrame(columns=[
                                "subject",
                                "experiment_type",
                                "id",
                                "prob",
                                "intervene_speed", # 0.0, 0.5, 0.8 thresh of recognition system
                                "display_frame",
                                "intervene_frame",
                                "critical_point",
                                "intervene_type",
                                ])



attrib_db = {}
attrib = getXmlRoot("/home/kuriatsu/Documents/experiment_data/pie_dataset/annotations_attributes/set04/video_0001_attributes.xml")
for pedestrian in attrib:
    attrib_db[pedestrian.get("id")] = pedestrian.get("critical_point")

attrib = getXmlRoot("/home/kuriatsu/Documents/experiment_data/pie_dataset/annotations_attributes/set04/video_0002_attributes.xml")
for pedestrian in attrib:
    attrib_db[pedestrian.get("id")] = pedestrian.get("critical_point")

# get data
data_path = "/home/kuriatsu/Dropbox/data/PIE_experiment_june/data"
for file in glob.glob(os.path.join(data_path, "*.csv")):
    data = pd.read_csv(file)
    filename =file.split("/")[-1]
    subject = filename.split("_", 1)[0]
    type = filename.rsplit("_", 1)[-1].replace(".csv", "")
    experiment_type = "BUTTON" if type == "enter" else "TOUCH"
    for i, row in data.iterrows():
        if not math.isnan(row.intervene_frame):
            intervene_time_frame = int(row.intervene_frame) - int(row.display_frame)
            intervene_time = float(row.intervene_time) - float(row.display_time)
        else:
            intervene_time = intervene_time_frame = math.nan

        buf = pd.Series([
            subject,
            experiment_type,
            row.id,
            row.prob,
            intervene_time,
            row.display_frame,
            row.intervene_frame,
            attrib_db.get(row.id),
            row.intervene_type,
            ], index=extracted_data.columns)

        extracted_data = extracted_data.append(buf, ignore_index=True)

extracted_data.to_csv(data_path+"/summary.csv")
