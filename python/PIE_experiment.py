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
        self.display_obj = [[0]]

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


    def manageDisplayObjects(self, frame_num, is_checked):
        display_obj = []
        min_index = None
        min_time = 20000

        # print('update start : ', frame_num)
        for obj_id in self.pie_data[frame_num]:
            if self.pie_data[frame_num][obj_id]['label'] == 'pedestrian':
                obj_info = self.pie_data[frame_num][obj_id]
                flag = 0

                for obj in self.display_obj:
                    # if already displayed, keep information
                    if obj[0] == obj_id:
                        # if checked and find important object, add checked flag and unimportantize
                        if obj[1] and is_checked:
                            obj[2] = True
                            obj[1] = False
                        # add
                        display_obj.append(obj)
                        flag = 1
                        continue

                # if the obj is new
                if flag == 0:
                    display_obj.append([obj_id, False, False, False]) # id, is_critical, is_checked, is_passed

                # if the obj is passed one
                if int(obj_info['critical_point']) < int(frame_num):
                    display_obj[-1][1] = False
                    display_obj[-1][2] = False

                # if the obj should be considered and unchecked, find the next important obj
                elif int(obj_info['critical_point']) < min_time and not display_obj[-1][2]:
                    min_index = len(display_obj) - 1
                    min_time = int(obj_info['critical_point'])

        # if the important obj is found, add the flag
        if min_index is not None:
            display_obj[min_index][1] = True

        # reflesh member container
        self.display_obj = display_obj


    def drawRect(self, image, frame_num):

        get_important_obj = False

        for obj in self.display_obj:
            if obj[1]: # if important --- red
                color = (0, 0, 255)

            elif obj[2]: # if checked --- green
                color = (255, 0, 0)

            else: # else --- blue
                color = (0, 255, 0)

            obj_info = self.pie_data[str(frame_num)][obj[0]]
            # print(obj_info, color)
            image = cv2.rectangle(image,
                                  (int(float(obj_info['xtl'])), int(float(obj_info['ytl']))),
                                  (int(float(obj_info['xbr'])), int(float(obj_info['ybr']))),
                                  color,
                                  1)


    def loop(self):
        print('start_loop')
        frame_num = 0
        sleep_time = self.video_rate
        is_checked = False

        while(self.video.isOpened()):

            start = time.time()

            ret, frame = self.video.read()

            if str(frame_num) in self.pie_data:
                self.manageDisplayObjects(str(frame_num), is_checked)
                self.drawRect(frame, frame_num)

            cv2.imshow('frame', frame)

            sleep_time = max(int((1 / self.video_rate - (time.time() - start)) * 1000), 1)

            is_checked = False
            # sleep and wait quit key
            key = cv2.waitKey(sleep_time) & 0xFF
            if key is not 255 : print(key)
            if key == ord('q'):
                break
            if key == 13:
                is_checked = True

            frame_num += 1

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
    # print(pie_data_visualize.pie_data.get('1133'))
    pie_data_visualize.loop()

if __name__ == '__main__':
    main()
