#! /usr/bin/python3
# -*- coding:utf-8 -*-

import csv
import xml.etree.ElementTree as ET
import argparse


def readCsv(filename):
    with open(filename, 'r') as file_obj:
        reader = csv.reader(file_obj)
        header = next(reader)
        return [row for row in reader]


def readXml(filename):
    try:
        tree = ET.parse(filename)
        root = tree.getroot()
    except:
        print('cannot open ', filename)
        exit(0)

    del tree
    return root


def alignData(result, annotations_tree, ego_vehicle_tree, attributes_tree):

    aligned_data_pedestrian = []
    aligned_data_tr = []

    for result_row in result:
        row = result_row

        if row[3] == 'pedestrian':
            for pedestrian in attributes_tree.iter('pedestrian'):
                if pedestrian.attrib.get('id') == row[2]:
                    row += [
                        pedestrian.attrib.get('age'),
                        pedestrian.attrib.get('critical_point'),
                        pedestrian.attrib.get('crossing_point'),
                        pedestrian.attrib.get('exp_start_point'),
                        pedestrian.attrib.get('gender'),
                        pedestrian.attrib.get('intersection'),
                        pedestrian.attrib.get('num_lanes'),
                        pedestrian.attrib.get('signalized'),
                        pedestrian.attrib.get('traffic_direction')
                        ]
                    break

            for box in annotations_tree.iter('box'):
                # print(box.attrib.get('frame'))
                for box_attribute in box:
                    if box_attribute.attrib.get('name') == 'id': id = box_attribute.text
                    if box_attribute.attrib.get('name') == 'action': action = box_attribute.text
                    if box_attribute.attrib.get('name') == 'cross': cross = box_attribute.text

                if int(box.attrib.get('frame')) == int(row[0]) and id == row[2]:
                    row += [
                        box.attrib.get('xtl'),
                        box.attrib.get('ytl'),
                        box.attrib.get('xbr'),
                        box.attrib.get('ybr'),
                        cross,
                        action
                    ]
                    break

            distance = 0
            for ego_vehicle in ego_vehicle_tree[int(row[0]):int(row[12])]:
                distance += float(ego_vehicle.attrib.get('GPS_speed'))  * 0.03 / 3.6

            row += [
                ego_vehicle_tree[int(row[0])].attrib.get('GPS_speed'),
                distance
            ]

            distance = 0
            for ego_vehicle in ego_vehicle_tree[int(row[7]):int(row[12])]:
                distance += float(ego_vehicle.attrib.get('GPS_speed'))  * 0.03 / 3.6

            row += [
                ego_vehicle_tree[int(row[7])].attrib.get('GPS_speed'),
                distance
            ]

            aligned_data_pedestrian.append(row)

        elif row[3] == 'traffic_light':
            for box in annotations_tree.iter('box'):
                if int(box.attrib.get('frame')) == row[0] and box[1].text == row[2]:
                    row += [
                        box.attrib.get('xtl'),
                        box.attrib.get('ytl'),
                        box.attrib.get('xbr'),
                        box.attrib.get('ybr'),
                        box[0].text,
                        box[2].text
                    ]
                    break

            aligned_data_tr.append(row)

    return aligned_data_pedestrian, aligned_data_tr


def writeCsv(pedestrian_data, tr_data, filename):

    print('pedestrian\n', pedestrian_data)
    print('tr\n', tr_data)

    with open(filename, 'w') as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(['displayed_frame', 'display_time', 'id', 'obj_type', 'frameout_frame', 'prob', 'intervene_type', 'intervene_frame', 'intervene_time', 'intervene_key',
                         'age', 'critical_point', 'crossing_point', 'exp_start_point', 'gender', 'intersection', 'num_lanes', 'signalized', 'traffic_direction',
                         'xtl', 'ytl', 'xbr', 'ybr', 'is_crossing', 'action',
                         'ego_speed_display', 'distance_display', 'ego_speed_intervene', 'distance_intervene'])

        writer.writerows(pedestrian_data)
        writer.writerow(['display_frame', 'display_time', 'id', 'obj_type', 'frameout_frame', 'prob', 'intervene_type', 'intervene_time', 'intervene_frame', 'intervene_key',
                         'xtl', 'ytl', 'xbr', 'ybr', 'type', 'state'])

        writer.writerows(tr_data)


def main():

    argparser = argparse.ArgumentParser( description = __doc__)
    argparser.add_argument(
    '--result', '-r',
    metavar='/path/to/result.csv',
    default='/home/kuriatsu/share/PIE_result/200529020823/kuriatsu_2.csv'
    )
    argparser.add_argument(
    '--annotations', '-anno',
    metavar='/path/to/annotations.xml',
    default='/media/ssd/PIE_data/annotations/set04/video_0006_annt.xml'
    )
    argparser.add_argument(
    '--ego_vehicle', '-v',
    metavar='/path/to/annotations_vehicle.xml',
    default='/media/ssd/PIE_data/annotations_vehicle/set04/video_0006_obd.xml'
    )
    argparser.add_argument(
    '--attributes', '-attr',
    metavar='/path/to/annotations_attributes.xml',
    default='/media/ssd/PIE_data/annotations_attributes/set04/video_0006_attributes.xml'
    )
    argparser.add_argument(
    '--output', '-o',
    metavar='/path/to/out.csv',
    default='/home/kuriatsu/share/PIE_result/200529020823/kuriatsu_2_aligned.csv'
    )
    args = argparser.parse_args()

    result = readCsv(args.result)
    annotations = readXml(args.annotations)
    ego_vehicle = readXml(args.ego_vehicle)
    attributes = readXml(args.attributes)
    aligned_data_pedestrian, aligned_data_tr = alignData(result, annotations, ego_vehicle, attributes)
    writeCsv(aligned_data_pedestrian, aligned_data_tr, args.output)

if __name__ == '__main__':

    main()
