#! /usr/bin/env python3
# -*- coding:utf-8 -*-

class PieLib():

    def getVideo(self, filename, rate_additional, image_offset_y, crop_rate):

         try:
             video = cv2.VideoCapture(args.video)

         except:
             print('cannot open video')
             exit(0)

         # get video rate and change variable unit from time to frame num
         frame_rate = int(self.video.get(cv2.CAP_PROP_FPS)) + rate_additional
         image_res = [video.get(cv2.CAP_PROP_FRAME_HEIGHT), video.get(cv2.CAP_PROP_FRAME_WIDTH)]
         # adjust video rate to keep genuine broadcast rate

         # calc image-crop-region crop -> expaned to original frame geometry
         offset_yt = image_res[0] * ((1.0 - crop_rate) * 0.5 + image_offset_y)
         offset_xl = image_res[1] * (1.0 - crop_rate) * 0.5
         crop_value = [int(offset_yt),
                      int(offset_yt + image_res[0] * crop_rate),
                      int(offset_xl),
                      int(offset_xl + image_res[1] * crop_rate)
                      ]

        return video, crop_value


    def getXmlRoot(self, filename):

        try:
            tree = ET.parse(filename)
            return tree.getroot()
            del tree
        except:
            print(f'cannot open {filename} file')
            exit(0)
