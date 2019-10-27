#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
import sys
from lxml import etree
import pykml
from pykml.parser import fromstring as from_kml_string
from pykml.parser import Schema
from pykml.factory import KML_ElementMaker as KML
from pykml.factory import GX_ElementMaker as GX

import csv


def getdata(filename):
    file = open(filename, 'r')

    reader = csv.reader(file)
    header = next(reader)

    # data = []
    point_string = str()
    dir_list = []
    vel_list = []

    for row in reader:
        # data.append(row)
        point_string += ','.join(map(str, row[:3]))
        point_string += ' '
        dir_list.append(row[3])
        vel_list.append(row[4])

    file.close()

    print(point_string)

    return point_string, dir_list, vel_list


def empty_folder(name):

    doc = KML.Folder(
            KML.name(name),
            KML.Open("0")
        )

    return doc


def make_lane_description(dir_list, vel_list):

    point_num = len(dir_list)

    description = 'WPID_1_AC_F_0_From_To_2_Lid_0_Rid_0_Vel_' + \
                  vel_list[0] + \
                  "_Dir_" + \
                  dir_list[0] + \
                  ","

    for i, dir in enumerate(dir_list[1:-2], 2):

        description += "WPID_" + \
                       str(i) + \
                       "_ACF_0_From_" + \
                       str(i-1) + \
                       "_To_" + \
                       str(i+1) + \
                       "_Lid_0_Rid_0_Vel_" + \
                       vel_list[i] + \
                       "_Dir_" + \
                       dir + \
                       ","

    description += "WPID_" + \
                   str(point_num+1) + \
                   "_ACF_0_From_" + \
                   str(point_num) + \
                   "_To_" + \
                   str(point_num) + \
                   "_Lid_0_Rid_0_Vel_" + \
                   vel_list[point_num-1] + \
                   "_Dir_" + \
                   dir_list[point_num-1] + \
                   ","

    return description


def write_kml(coord, dir, vel, filename):

    doc = KML.kml(
        KML.Folder(
            KML.name("OpenPlanner KML based Map"),
            KML.Open("0"),
            KML.Document(
                KML.name("Map Data"),
                KML.Open("0"),
                empty_folder("CurbsLines"),
                empty_folder("Boundaries"),
                empty_folder("Markings"),
                empty_folder("Crossings"),
                empty_folder("TrafficSigns"),
                empty_folder("TrafficLights"),
                empty_folder("StopLines"),
                empty_folder("RoadSegments"),
                KML.Folder(
                    KML.name("Lanes"),
                    KML.Open("0"),
                    KML.Folder(
                        KML.name("LID_1"),
                        KML.Open("0"),
                        KML.description("LID_1_RSID_1_NUM_0_From_-1_To_1_Vel_3"),
                        KML.Placemark(
                            KML.name("waypoints"),
                            KML.Open("0"),
                            KML.LineString(
                                KML.coordinates(coord)
                            )
                        ),
                        KML.Folder(
                            KML.description(make_lane_description(dir, vel))
                        ),
                    ),
                ),
            )
        )
    )

    print(type(doc))
    # print(etree.tostring(doc, pretty_print=True))

    # output a KML file (named based on the Python script)

    filename = filename.rstrip('.csv')+'.kml'
    outfile = open(filename, 'w')

    outfile.write(str("<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n"))
    outfile.write(etree.tostring(doc, pretty_print=True).decode('utf-8'))


if __name__ == "__main__":

    args = sys.argv

    coord, dir, vel = getdata(args[1])
    write_kml(coord, dir, vel, args[1])
