#!/usr/bin/python

import xml.etree.ElementTree as ET
import argparse
import os.path
import sys

def _validator_file_exists(parser, arg):
    if not os.path.exists(arg):
        parser.error('The file "{}" does not exist!'.format(arg))
    return arg


def parse_xml_annotation(filename):
    try:
        xml_tree = ET.parse(filename)
        xml_root = xml_tree.getroot()
        if xml_root.tag != 'annotation':
            return False, 'Parse error: File is not a valid annotation file'

        # Find all annotated objects
        xml_objects = xml_root.findall('object')
        if len(xml_objects) == 0:
            return False, 'Parse Error: There is no annotation'

        annotations = [{} for _ in xml_objects]

        # Parse all annotated objects
        object_id = 0
        for xml_object in xml_objects:
            xml_polygons = xml_object.findall('polygon')

            for xml_polygon in xml_polygons:
                frame_num = int(xml_polygon.findall('t')[0].text) + 1

                # Parse rectangle from polygon file
                xml_pts = xml_polygon.findall('pt')
                rect = [float('inf'), float('inf'), -float('inf'), -float('inf')]
                for xml_pt in xml_pts:
                    x = float(xml_pt.find('x').text)
                    y = float(xml_pt.find('y').text)
                    rect[0] = min(rect[0], x)
                    rect[1] = min(rect[1], y)
                    rect[2] = max(rect[2], x)
                    rect[3] = max(rect[3], y)

                annotations[object_id][frame_num] = rect

            object_id += 1

        return True, annotations
    except ET.ParseError:
        return False, 'Parse error: File "{}" is not a valid xml file'.format(filename)

    return False, 'Parse error'

class FileStreamWriter:
    def __init__(self, filename=None):
        self._filename = filename

    def __enter__(self):
        if self._filename is None:
            self._fp = None
            return sys.stdout
        else:
            self._fp = open(self._filename, 'w')
            return self._fp

    def __exit__(self, type, value, traceback):
        if self._fp is not None:
            self._fp.close()

def main():
    description = 'Simple script that converts xml formatted annotation file into mot format'
    arg_parser = argparse.ArgumentParser(description=description)
    arg_parser.add_argument("src", type=lambda x: _validator_file_exists(arg_parser, x),
                            help='Input xml file ( Must be Vatic compatible file )')
    arg_parser.add_argument("dst", nargs='?', default=None,
                            help='Destination file (if not present outputs into stdout)')

    args = arg_parser.parse_args()

    success, result = parse_xml_annotation(args.src)

    if success is False:
        error = result
        print(error)
        return
    else:
        annotations = result
        start_frame = sys.maxsize
        end_frame = -sys.maxsize

        # Find start_frame, end_frame numbers
        for annotation in annotations:
            end_frame = max(end_frame, max(annotation.keys()))
            start_frame = min(start_frame, min(annotation.keys()))

        # Write to file (or stdout)
        with FileStreamWriter(args.dst) as fp:
            for frame_num in range(start_frame, end_frame + 1):
                for i in range(len(annotations)):
                    object_id = i + 1
                    if frame_num in annotations[i]:
                        rect = annotations[i][frame_num]
                        values = (frame_num, object_id, rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1])
                        fp.write("{},{},{},{},{},{},1,-1,-1,-1\n".format(*values))


if __name__ == "__main__":
    main()
