#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
welcome to PIE data visualize

"""

import cv2
import argparse
import numpy as np
import time
import xml.etree.ElementTree as ET


class PieDataVisualize(object):

    def __init__(self, args):
        self.video = None
        self.video_fps = None
        self.attrib_tree = None
        self.pie_data = {}

    def getVideo(self, video_file):
        try:
            self.video = cv2.VideoCapture(video_file)
            self.video_rate = int(self.video.get(cv2.CAP_PROP_FPS))+5
        except:
            print('cannot open video')
            exit(0)


    def getAttrib(self, attrib_file):
        try:
            tree = ET.parse(attrib_file)
            self.attrib_tree = tree.getroot()
            del tree
        except:
            print('cannot open attrib file')
            exit(0)


    def getAnno(self, anno_file):
        try:
            tree = ET.parse(anno_file)
            root = tree.getroot()
        except:
            print('cannot open annotation file')
            exit(0)

        for track in root.findall('track'):

            for anno_itr in track.iter('box'):

                anno_info = {}

                # get id
                for attribute in anno_itr.findall('attribute'):
                    if attribute.attrib.get('name') == 'id':
                        anno_id = attribute.text

                # get basic information
                anno_info['label'] = track.attrib.get('label')
                anno_info['xbr'] = anno_itr.attrib.get('xbr')
                anno_info['xtl'] = anno_itr.attrib.get('xtl')
                anno_info['ybr'] = anno_itr.attrib.get('ybr')
                anno_info['ytl'] = anno_itr.attrib.get('ytl')

                # if object is pedestrian, get additional information
                if anno_info['label'] == 'pedestrian':
                    for attrib_itr in self.attrib_tree.iter('pedestrian'):
                        if attrib_itr.attrib.get('id') == anno_id:
                            anno_info['intention_prob'] = attrib_itr.attrib.get('intention_prob')
                            anno_info['critical_point'] = attrib_itr.attrib.get('critical_point')
                            anno_info['crossing_point'] = attrib_itr.attrib.get('crossing_point')
                            anno_info['exp_start_point'] = attrib_itr.attrib.get('exp_start_point')

                # add to pie_data dictionary
                if anno_itr.attrib.get('frame') not in self.pie_data:
                    self.pie_data[anno_itr.attrib.get('frame')] = {}

                self.pie_data[anno_itr.attrib.get('frame')][anno_id] = anno_info

        # delete objects to improve performance
        del root
        del tree
        del self.attrib_tree

    def drawRect(self, image, frame_num):
        if frame_num in self.pie_data:
            for obj in self.pie_data[frame_num].values():
                if obj.get('label') == 'pedestrian':
                    image = cv2.rectangle(image,
                                          (int(float(obj.get('xtl'))), int(float(obj.get('ytl')))),
                                          (int(float(obj.get('xbr'))), int(float(obj.get('ybr')))),
                                          (0, 255, 0),
                                          3)


    def loop(self):
        print('start_loop')
        frame_num = 0
        sleep_time = self.video_rate
        while(self.video.isOpened()):

            start = time.time()
            ret, frame = self.video.read()
            self.drawRect(frame, str(frame_num))
            cv2.imshow('frame', frame)
            frame_num += 1

            sleep_time = max(int((1 / self.video_rate - (time.time() - start)) * 1000), 1)
            # sleep and wait quit key
            if cv2.waitKey(sleep_time) & 0xFF == ord('q'):
                break


    def __del__(self):
        print('delete instance...')
        self.video.release()
        cv2.destroyAllWindows()


def main():
    argparser = argparse.ArgumentParser( description = __doc__)
    argparser.add_argument(
        '--video', '-v',
        metavar='VIDEO',
        default='/media/ssd/PIE_data/PIE_clips/set01/video_0001.mp4')
    argparser.add_argument(
        '--anno',
        metavar='ANNO',
        default='/media/ssd/PIE_data/annotations/set01/video_0001_annt.xml')
    argparser.add_argument(
        '--attrib',
        metavar='ATTRIB',
        default='/media/ssd/PIE_data/annotations_attributes/set01/video_0001_attributes.xml')

    args = argparser.parse_args()

    pie_data_visualize = PieDataVisualize(args)
    pie_data_visualize.getVideo(args.video)
    pie_data_visualize.getAttrib(args.attrib)
    pie_data_visualize.getAnno(args.anno)
    pie_data_visualize.getAttrib(args.attrib)
    print(pie_data_visualize.pie_data.get('1133'))
    pie_data_visualize.loop()

if __name__ == '__main__':
    main()
