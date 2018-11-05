# /usr/bin/env python
# -*- coding: utf-8 -*-

import logging

logger = logging.getLogger(__name__)
import json
import numpy as np
import random
from scipy.spatial import Delaunay
from skimage import morphology
from collections import OrderedDict

description = {}


def get_segdata(json_data, datap, labels=None):
    Z = len(datap["segmentation"])
    X = len(datap["segmentation"][0])
    Y = len(datap["segmentation"][0][0])

    for slice in range(0, Z):
        nbr_drawings = json_data['drawings'][slice][0]['length']
        for draw in range(0, nbr_drawings):
            # zpracovavany label
            dict_key = json_data['drawingsDetails'][slice][0][draw]['textExpr']
            if labels != None and dict_key not in labels: # zpracovavani jen nekterych objektu
                continue

            # riznuti jsonu u souradnic krajnich bodu
            draw_info = json_data['drawings'][slice][0][str(draw)]
            start = draw_info.find("points")
            end = draw_info.find("stroke")
            points_data = draw_info[start + 9: end - 3].split(',')     

            # prepsani souradnic do pole
            len_array = int(len(points_data) / 2)
            points = np.empty((len_array, 2))
            i = 0
            for j in range(0, len_array):
                points[j] = [int(points_data[i + 1]), int(points_data[i])]
                i += 2

            # vyplneni nakresleneho obrazce
            # pri spatnem nakresleni krivky muze zde padnout
            hull = Delaunay(points)
            x, y = np.mgrid[0:X, 0:Y]
            grid = np.vstack([x.ravel(), y.ravel()]).T
            simplex = hull.find_simplex(grid)
            fill = grid[simplex >= 0, :]

            # vlozeni markeru do dat
            # pokud je label ve slovniku, pouzije se hodnota z nej, neprepise se na novou!
            # pokud label neni ve slovniku, priradi se mu nahodna hodnota v rozmezi 100 - 254, 
            # jestlize neobsahuje v popisu vlastni hodnotu
            # pokud neni uveden label, ale pouze hodnota, vytvori se label se strukturou "lbl_" + hodnota
            # jestlize neni uveden ani label, ani hodnota, nic se neprovede
            dict_value = 0
            dict_description = json_data['drawingsDetails'][slice][0][draw]['longText'].replace("'", '"')
            if dict_description == '':
                if dict_key == '':
                    print("Drawing is not defined at slice", slice)
                    continue
                else:
                    if dict_key in datap["slab"].keys():
                        dict_value = datap["slab"][dict_key]
                    else:
                        dict_value = random.randint(100, 254)
                        datap["slab"][dict_key] = dict_value
            else:
                dict_description = json.loads(dict_description)
                if dict_key != '' and dict_key in datap["slab"].keys():
                    dict_value = datap["slab"][dict_key]
                elif "value" in dict_description.keys():
                    dict_value = dict_description["value"]
                    if dict_key != '':
                        datap["slab"][dict_key] = dict_value
                    else:
                        dict_key = "lbl_" + str(dict_value)
                        datap["slab"][dict_key] = dict_value
                elif dict_key != '':
                    dict_value = random.randint(100, 254)
                    datap["slab"][dict_key] = dict_value
                else:
                    print("Drawing is not defined at slice", slice)
                    continue

            for i, j in fill:
                datap["segmentation"][Z - 1 - slice][i][j] = dict_value
        
            # ziskani zbytku popisu
            if dict_key not in description.keys():
                description[dict_key] = {}
                i_color = draw_info.find('#') + 1
                description[dict_key]["r"] = int(draw_info[i_color:(i_color + 2)], 16)
                description[dict_key]["g"] = int(draw_info[(i_color + 2):(i_color + 4)], 16)
                description[dict_key]["b"] = int(draw_info[(i_color + 4):(i_color + 6)], 16)
                description[dict_key]["value"] = dict_value

            if dict_description != '':
                if "threshold" in dict_description.keys():
                    description[dict_key]["threshold"] = dict_description["threshold"] # nastavit kolem 100 - 120
                if "two" in dict_description.keys():
                    description[dict_key]["two"] = dict_description["two"]
                if "three" in dict_description.keys():
                    description[dict_key]["three"] = dict_description["three"]


def get_vesselpoint_in_seeds(json_data, label, seeds_shape):
    seeds = np.zeros(seeds_shape, dtype=int)
    pts = get_vesselPoint(json_data, label)
    seeds[pts] = 1
    return seeds


def get_vesselPoint(json_data, label=None):
    Z = len(json_data['drawings'])

    if label==None:
        label = "porta"

    for slice in range(0, Z):
        nbr_drawings = json_data['drawings'][slice][0]['length']
        for draw in range(0, nbr_drawings):
            # zpracovavany label
            dict_key = json_data['drawingsDetails'][slice][0][draw]['textExpr']
            if dict_key != label:
                continue

            # riznuti jsonu u souradnic krajnich bodu
            draw_info = json_data['drawings'][slice][0][str(draw)]
            start = draw_info.find("points")
            end = draw_info.find("stroke")
            points_data = draw_info[start + 9: end - 3].split(',')     

            # prepsani souradnic do pole
            z = np.array([(Z - 1 - slice)], dtype=np.dtype('int64'))
            x = np.array([int(points_data[1])], dtype=np.dtype('int64'))
            y = np.array([int(points_data[0])], dtype=np.dtype('int64'))
            return [z, x, y]


def get_seeds(data, label):
    return ((data["segmentation"] != 0).astype('int8') * 2 - 
           (data["segmentation"] == data["slab"][label]).astype('int8'))


def write_to_json(datap, json_data=None, output_name="json_data.json"):
    Z = len(datap["segmentation"])
    X = len(datap["segmentation"][0])
    Y = len(datap["segmentation"][0][0])

    if json_data==None:
        json_data = initJson(Z) # prohazuje poradi => dwv nefunguje na 100 % (nelze pak stahnout json)

    for slice in range(0, Z):
        label_array = np.unique(datap["segmentation"][Z - 1 - slice]).tolist()
        label_array.remove(0) # pole vyskytovanych labelu bez 0

        divided = morphology.label(datap["segmentation"][Z - 1 - slice], background=0) # rozdeleni objektu
        nbr_divided = np.unique(divided).tolist() 
        nbr_divided.remove(0) # pole rozdelenych objektu bez 0
        logger.debug("slice, label array, number divided: {} {} {}".format(slice, label_array, nbr_divided))

        if len(label_array) < len(nbr_divided): # nesedi pocet vyskytovanych labelu s poctem objektu
            new_array = []
            for i in range(0, max(nbr_divided)):
                for j in range(0, len(label_array)):
                    a = ((datap["segmentation"] == label_array[j]) * (divided == i)).astype('int8')
                    if 1 in a:
                        new_array.append(label_array[j])
                        break
            label_array = new_array
        if len(label_array) > 0:
            nbr_drawings = json_data['drawings'][slice][0]['length']
            for lbl in range(0, len(label_array)):
                str_points = ""
                nbr_points = 0
                last_y = 0
                last_x = 0
                for key, value in datap["slab"].items(): # neni osetreno, kdyby se nenachazel label ve slovniku
                    if value != 0 and value == label_array[lbl]:
                        for x in range(0, X):
                            for y in range(0, Y):
                                if divided[x][y] == lbl + 1:
                                    str_points += ("," if len(str_points) != 0 else "") + str(y) + "," + str(x)
                                    last_y = y
                                    last_x = x
                                    nbr_points += 1
                        break
                if nbr_points <= 1: # osetreni, aby DWV nezamrzl (nesmi byt predan 1 stejny bod)
                    str_points += "," + str(last_y + 1) + "," + str(last_x)
                if key not in description.keys():
                    initDescription(key, b=0)

                rgba = "rgba(" + str(description[key]["r"])
                rgba += "," + str(description[key]["g"]) + ","
                rgba += str(description[key]["b"]) + ",0.5)"
                json_data["drawings"][slice][0][str(lbl + nbr_drawings)] = get_str_drawings(key + str(lbl), key, str_points, rgba, [150, 10 + lbl * 12])
                json_data["drawingsDetails"][slice][0].append({"id":key + str(lbl), "textExpr":key, "longText":"{\"value\":" + str(value) + "}", "quant":None})
            json_data["drawings"][slice][0]["length"] = len(label_array) + nbr_drawings
    with open(output_name, 'w') as file:
        json.dump(json_data, file)

def initJson(nbr_slices, window_center=50, window_width=350, y=0, x=0, z=0, scale=1):
    json_data = OrderedDict()
    json_data["version"] = "0.2"
    json_data["window-center"] = window_center
    json_data["window-width"] = window_width
    json_data["position"] = OrderedDict([("i", y), ("j", x), ("k", z)])
    json_data["scale"] = scale
    json_data["scaleCenter"] = OrderedDict([("x",0), ("y", 0)])
    json_data["translation"] = OrderedDict([("x",0), ("y", 0)])
    json_data["drawings"] = []
    json_data["drawingsDetails"] = []
    for slice in range(0, nbr_slices):
        json_data["drawings"].append([{"length":0}])
        json_data["drawingsDetails"].append([[]])
    return json_data

def initDescription(label, value=None, r=150, g=150, b=150):
    description[label] = {}
    description[label]["r"] = r
    description[label]["g"] = g
    description[label]["b"] = b
    if value==None:
        value = 93
    description[label]["value"] = value
    

def get_str_drawings(id, key, str_points, color, lbl_pos):
    string = "{\"attrs\":{\"name\":\"freeHand-group\",\"visible\":true,\"id\":\"" + id +"\"},"
    string += "\"className\":\"Group\",\"children\":[{\"attrs\":{\"points\":[" + str_points + "],"
    string += "\"stroke\":\"" + color + "\",\"strokeWidth\":2,\"name\":\"shape\",\"tension\":0.5,"
    string += "\"draggable\":true},\"className\":\"Line\"},"
    string += "{\"attrs\":{\"x\":" + str(lbl_pos[0]) + ",\"y\":" + str(lbl_pos[1]) + ",\"name\":\"label\"},"
    string += "\"className\":\"Label\",\"children\":[{\"attrs\":{\"fontSize\":11,\"fontFamily\":\"Verdana\","
    string += "\"fill\":\"" + color + "\",\"name\":\"text\",\"text\":\"" + key + "\"},"
    string += "\"className\":\"Text\"},{\"attrs\":{\"width\":24,\"height\":12},\"className\":\"Tag\"}]}]}"
    return string