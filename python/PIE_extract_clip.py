#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import cv2
import xml.etree.ElementTree as ET
import numpy as np
from pie_lib import PieLib
import glob
import os
import random
import pickle

def pieExtractClip(video, crop_value, crop_rate, attrib_tree, annt_tree, out_data):

    image_res = [video.get(cv2.CAP_PROP_FRAME_HEIGHT), video.get(cv2.CAP_PROP_FRAME_WIDTH)]
    frame_rate = int(self.video.get(cv2.CAP_PROP_FPS))
    expand_rate = 1.0 / crop_rate
    for attrib in attrib_tree.iter('pedestrian'):
        out_data[attrib.attrib.get('id')] = {
            'label' : 'pedestrian',
            'prob' : attrib.attrib.get('prob'),
            'critical_point' : attrib.attrib.get('critical_point'),
            'crossing_point' : attrib.attrib.get('crossing_point'),
            'start_point' : attrib.attrib.get('exp_start_point')
            }

    for track in annt_tree.iter('track'):
        label = track.attrib.get('label')

        if label not in ['pedestrian', 'traffic_light']: continue

        for annt_attrib in track[0].findall('attribute'):
            if annt_attrib.attrib.get('name') == 'id':
                id = annt_attrib.text

        if label == 'traffic_light':
            out_data[id] = {
                'label' : label,
                'prob' : random.random(),
                'critical_point' : int(track[-1].attrib.get('frame')),
                'crossing_point' : int(track[-1].attrib.get('frame')),
                'start_point' : int(float(track[-1].attrib.get('frame')) - random.random() * 3.0 * frame_rate)
            }

        frame_info_list = []
        for annt_itr in track.iter('box'):
            frame_info = {}
            frame_index = int(annt_itr.attrib.get('frame_index'))

            if frame_index < put_data.get(id).get('start_point'):
                continue
            elif put_data.get(id).get('critical_point') < frame_index:
                break

            frame_info['xbr'] = int((float(annt_itr.attrib.get('xbr')) - crop_value[2]) * (1 / crop_rate))
            frame_info['xtl'] = int((float(annt_itr.attrib.get('xtl')) - crop_value[2]) * (1 / crop_rate))
            frame_info['ybr'] = int((float(annt_itr.attrib.get('ybr')) - crop_value[0]) * (1 / crop_rate))
            frame_info['ytl'] = int((float(annt_itr.attrib.get('ytl')) - crop_value[0]) * (1 / crop_rate))

            if xtl < 0 or xbr > image_res[1] or ytl < 0 or ybr > image_res[0]:
                out_data.get(id)['critical_point'] = frame_index - 1
                break

            frame_info_list.append(frame_info)

        frame_list = []
        for index in range(out_data.get(id), out_data.get('critical_point')):
            video.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = video.read()
            if ret:
                frame = cv2.resize(frame[crop_value[0]:crop_value[1], crop_value[2]:crop_value[3]], dsize=None, fx=expand_rate, fy=expand_rate)
                frame_list.append(frame)
            else:
                break

        out_data.get('id')['frames'] = frame_list
        out_data.get('id')['frames_info'] = frame_info_list



def main(args):

    out_data = {}

    for video_file in glob.iglob(args.video_dir+'/set*/*.mp4'):
        set_name = video_file.split('/')[-2]
        video_name = video_file.split('/')[-1].split('.')[-2]
        attrib_file = args.attrib_dir+'/'+set_name+'/'+video_name+'_attributes.xml'
        annt_file = args.anno_dir+'/'+set_name+'/'+video_name+'_annt.xml'

        if os.path.isfile(attrib_file) and os.path.isfile(annt_file):

            video, crop_value = PieLib.getVideo(video_file, args.image_offset_y, args.crop_rate)
            attrib = PieLib.getXmlRoot(attrib_file)
            annt = PieLib.getXmlRoot(annt_file)

            pieExtractClip(video, crop_value, args.crop_rate, attrib, annt, out_data)

        with open(args.out_file, mode='wb') as file:
            pickle.dump(out_data, file)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser( descriotion = __doc__)
    argparser.add_argument(
        '--video_dir',
        default='/media/ssd/PIE_data/PIE_clips',
        metavar='DIR')
    argparser.add_argument(
        '--annotation_dir',
        default='/media/ssd/PIE_data/annotations',
        metavar='DIR')
    argparser.add_argument(
        '--attrib_dir',
        default='/media/ssd/PIE_data/annotations_attributes',
        metavar='DIR')
    argparser.add_argument(
        '--crop_rate',
        metavar='SCALE',
        default=0.6)
    argparser.add_argument(
        '--image_offset_y',
        metavar='OFFSET',
        default=0.2)
    argparser.add_argument(
        '--out_file',
        metavar='DIR',
        default='/media/ssd/PIE_data/extracted_data.pickle')

    args = argparser.parse_args()

    main(args)
