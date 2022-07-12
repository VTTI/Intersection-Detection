import re
import xml.etree.ElementTree as ET
from ast import literal_eval


def get_points(polygon):
    points = polygon.attrib["points"].split(';')
    points_list = list()
    for point in points:
        points_list.append(literal_eval(point))
    return points_list, len(points_list)


def parse_xml(xmlfile):
    try:
        tree = ET.parse(xmlfile)  # create element tree object
        # get root element
        root = tree.getroot()
        for image in root.findall("image"):
            image_name = image.attrib["name"]
            unique_image_identifier = re.findall("_([a-z0-9]+)_", image_name)[0]
            for polygon in image.findall("polygon"):
                label = polygon.attrib["label"]
                points, _ = get_points(polygon)
                # search for workzone related objects
                if label == "intersection":
                    if polygon.find("attribute").text == "signalized_intersection":
                        print("signalized_intersection", points)
                    if polygon.find("attribute").text == "signalized_intersection":
                        print("non signalized", points)


    except Exception as e:
        print(e)


if __name__ == "__main__":
    pass
