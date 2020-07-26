#! /usr/bin/python3
# -*- coding:utf-8 -*-

import csv
import xml.etree.ElementTree as ET
import argparse
import numpy as np
import glob

intervene_result_of_obj = {}
intervene_result_of_time = {}
aligned_data = []

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


def alignData(result, annotations_tree, ego_vehicle_tree, attributes_tree, experiment_intervene_type, video_start_time):

    out_pedestrian = []
    out_tr = []

    for result_row in result:
        # row = result_row
        out_row = []
        # print(out_row)
        display_point = result_row[0]
        obj_id = result_row[2]
        obj_type = result_row[3]
        prob = result_row[4]
        clarity = np.fabs(float(prob) - 0.5)
        intervene_type = result_row[7]
        time = float(display_point) / 30 + video_start_time
        experiment_type = 1 if experiment_intervene_type == 'touch' else 0


        if intervene_type in ['pushed', 'touched']:

            intervene_point = result_row[8]
            intervene_time = (float(intervene_point) - float(display_point)) / 30

            if float(prob) < 0.5:
                is_intervene_correct = 1
                answer_cross = 0
            else:
                is_intervene_correct = 0
                answer_cross = 0


        elif intervene_type == 'passed':
            intervene_time = None
            intervene_point = None

            if float(prob) >= 0.5:
                is_intervene_correct = 1
                answer_cross = 1
            else:
                is_intervene_correct = 0
                answer_cross = 1

        # intervene acc per obj
        if obj_id in intervene_result_of_obj:
            intervene_result_of_obj[obj_id]['total'] += 1
            intervene_result_of_obj[obj_id]['correct'] += is_intervene_correct
            intervene_result_of_obj[obj_id]['cross'] += answer_cross
            if experiment_type:
                intervene_result_of_obj[obj_id]['total_touch'] += 1
                intervene_result_of_obj[obj_id]['num_touch'] += answer_cross
                intervene_result_of_obj[obj_id]['correct_touch'] += is_intervene_correct
            else:
                intervene_result_of_obj[obj_id]['total_enter'] += 1
                intervene_result_of_obj[obj_id]['num_enter'] += answer_cross
                intervene_result_of_obj[obj_id]['correct_enter'] += is_intervene_correct
        else:
            intervene_result_of_obj[obj_id] = {'total':1,
                                               'correct':is_intervene_correct,
                                               'frame': display_point,
                                               'cross':answer_cross,
                                               'prob':prob,
                                               'clarity': clarity,
                                               'total_enter': 0,
                                               'num_enter':0,
                                               'correct_enter':0,
                                               'total_touch': 0,
                                               'num_touch':0,
                                               'correct_touch':0
                                               }
            if experiment_type:
                intervene_result_of_obj[obj_id]['total_touch'] += 1
                intervene_result_of_obj[obj_id]['num_touch'] += answer_cross
                intervene_result_of_obj[obj_id]['correct_touch'] += is_intervene_correct
            else:
                intervene_result_of_obj[obj_id]['total_enter'] += 1
                intervene_result_of_obj[obj_id]['num_enter'] += answer_cross
                intervene_result_of_obj[obj_id]['correct_enter'] += is_intervene_correct

        # intervene acc per time
        if int(time / 60) in intervene_result_of_time:
            intervene_result_of_time[int(time / 60)]['total'] += 1
            intervene_result_of_time[int(time / 60)]['correct'] += is_intervene_correct
            if experiment_type :
                intervene_result_of_time[int(time / 60)]['total_touch'] += 1
                intervene_result_of_time[int(time / 60)]['correct_touch'] += is_intervene_correct
            else:
                intervene_result_of_time[int(time / 60)]['total_enter'] += 1
                intervene_result_of_time[int(time / 60)]['correct_enter'] += is_intervene_correct

        else:
            # intervene_result_of_time[int(time / 60)] = {'total':1, 'correct':is_intervene_correct}
            intervene_result_of_time[int(time / 60)] = {'total':1, 'correct':is_intervene_correct, 'total_enter': 0, 'correct_enter':0, 'total_touch':0, 'correct_touch':0}
            if experiment_type :
                intervene_result_of_time[int(time / 60)]['total_touch'] = 1
                intervene_result_of_time[int(time / 60)]['correct_touch'] = is_intervene_correct
            else:
                intervene_result_of_time[int(time / 60)]['total_enter'] = 1
                intervene_result_of_time[int(time / 60)]['correct_enter'] = is_intervene_correct


        if obj_type == 'pedestrian':
            for pedestrian in attributes_tree.iter('pedestrian'):
                if pedestrian.attrib.get('id') == obj_id:
                    critical_point = pedestrian.attrib.get('critical_point')
                    crossing_point = pedestrian.attrib.get('crossing_point')
                    break

            for box in annotations_tree.iter('box'):
                # print(box.attrib.get('frame'))
                for box_attribute in box:
                    if box_attribute.attrib.get('name') == 'id': box_id = box_attribute.text
                    if box_attribute.attrib.get('name') == 'action': action = box_attribute.text

                if box.attrib.get('frame') == display_point and box_id == obj_id:
                    box_size = np.sqrt((int(float(box.attrib.get('xbr'))) - int(float(box.attrib.get('xtl')))) ** 2 + (int(float(box.attrib.get('ybr'))) - int(float(box.attrib.get('ytl')))) ** 2)
                    break

            # appear ego_vehicle info
            display_distance = 0
            for ego_vehicle in ego_vehicle_tree[ int(display_point) : int(crossing_point) ]:
                display_distance += float(ego_vehicle.attrib.get('GPS_speed'))  * 0.03 / 3.6

            display_ego_vel = float(ego_vehicle_tree[int(display_point)].attrib.get('GPS_speed'))

            # intervene ego_vehicle info
            if intervene_point is not None:
                intervene_distance = 0
                for ego_vehicle in ego_vehicle_tree[ int(intervene_point) : int(crossing_point) ]:
                    intervene_distance += float(ego_vehicle.attrib.get('GPS_speed'))  * 0.03 / 3.6

                intervene_ego_speed = float(ego_vehicle_tree[int(intervene_point)].attrib.get('GPS_speed'))

            else:
                intervene_distance = None
                intervene_ego_speed = None

            if intervene_point is not None:
                intervene_early = (float(critical_point)-float(intervene_point))/30
            else:
                intervene_early = None
            aligned_data.append([experiment_type, time, prob, clarity, display_distance, display_ego_vel, box_size, intervene_distance, intervene_ego_speed, intervene_time, is_intervene_correct, intervene_early])

        # elif obj_type == 'traffic_light':
        #     for box in annotations_tree.iter('box'):
        #         if int(box.attrib.get('frame')) == display_point and box[1].text == obj_id:
        #             out_row += [
        #                 box.attrib.get('xtl'),
        #                 box.attrib.get('ytl'),
        #                 box.attrib.get('xbr'),

        #                 box.attrib.get('ybr'),
        #                 box[0].text,
        #                 box[2].text
        #             ]
        #             break
        #
        #     out_tr.append(out_row)

    return out_pedestrian, out_tr


def writeCsv(pedestrian_data, filename):

    # print('pedestrian\n', pedestrian_data)

    with open(filename, 'a') as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(['intervene_type', 'time', 'prob', 'clarity', 'display_distance', 'display_velocity', 'box_size', 'intervene_distance', 'intervene_ego_speed', 'intervene_time', 'is_intervene_correct', 'critical_point'])

        writer.writerows(pedestrian_data)
        # writer.writerow(['display_point', 'display_time', 'id', 'obj_type', 'frameout_frame', 'prob', 'intervene_type', 'intervene_time', 'intervene_point', 'intervene_key',
        #                  'xtl', 'ytl', 'xbr', 'ybr', 'type', 'state'])

        # writer.writerows(tr_data)


def writeDict(obj, time, file):
    with open(file, 'a') as file_obj:
        writer = csv.writer(file_obj)

        writer.writerow(['id','frame', 'prob_anno', 'clarity', 'total', 'correct', 'enter_total', 'enter_num', 'correct_enter', 'touch_total', 'touch_num', 'correct_touch'])
        for key, val in obj.items():
            # acc = float(val['correct']) / float(val['total'])
            #
            # prob_experiment = float(val['cross']) / float(val['total'])
            # prob_enter = float(val['num_enter'])/ float(val['total_enter'])
            # prob_touch = float(val['num_touch'])/ float(val['total_touch'])]
            writer.writerow([key, val['frame'], val['prob'], val['clarity'], val['total'], val['correct'], val['total_enter'], val['num_enter'], val['correct_enter'], val['total_touch'], val['num_touch'], val['correct_touch']])

        writer.writerow(['minute', 'total', 'acc', 'acc_enter', 'acc_touch'])
        for key, val in time.items():
            writer.writerow([key, val['total'], float(val['correct']) / float(val['total']), float(val['correct_enter']) / float(val['total_enter']), float(val['correct_touch']) / float(val['total_touch'])])



def main():

    argparser = argparse.ArgumentParser( description = __doc__)
    argparser.add_argument(
    '--result', '-r',
    metavar='/path/to/result/',
    default='/home/kuriatsu/share/PIE_result/june/',
    # nargs = '*'
    )
    argparser.add_argument(
    '--annotations', '-anno',
    metavar='/path/to/annotations.xml',
    default='/media/ssd/PIE_data/annotations/set04/'
    # default='/media/ssd/PIE_data/annotations/set04/video_0001_annt.xml'
    )
    argparser.add_argument(
    '--ego_vehicle', '-v',
    metavar='/path/to/annotations_vehicle.xml',
    default='/media/ssd/PIE_data/annotations_vehicle/set04/'
    # default='/media/ssd/PIE_data/annotations_vehicle/set04/video_0001_obd.xml'
    )
    argparser.add_argument(
    '--attributes', '-attr',
    metavar='/path/to/annotations_attributes.xml',
    default='/media/ssd/PIE_data/annotations_attributes/set04/'
    # default='/media/ssd/PIE_data/annotations_attributes/set04/video_0001_attributes.xml'
    )

    args = argparser.parse_args()

    # aligned_data_pedestrian = []

    for result_file in glob.glob(args.result + '*'):
        # intervene_result_of_time.clear()
        # intervene_result_of_obj.clear()
        print(result_file)
        result = readCsv(result_file)
        annotations = readXml(args.annotations + 'video_000' + result_file.split('.')[0].split('_')[-2] + '_annt.xml')
        ego_vehicle = readXml(args.ego_vehicle + 'video_000' + result_file.split('.')[0].split('_')[-2] + '_obd.xml')
        attributes = readXml(args.attributes + "video_000" + result_file.split('.')[0].split('_')[-2] + "_attributes.xml")
        alignData(result, annotations, ego_vehicle, attributes, result_file.split('.')[0].split('_')[-1], (int(result_file.split('.')[0].split('_')[-2]) - 1) * 540)

    writeDict(intervene_result_of_obj, intervene_result_of_time, args.result + 'result_statistic.csv')
    writeCsv(aligned_data, args.result + 'result_logistic.csv')
    print(intervene_result_of_obj)
    print(intervene_result_of_time)

if __name__ == '__main__':

    main()
