import json
import numpy as np
from scipy.spatial import Delaunay

# data["slab"] = {"liver": 10, "hearth": 20, "left_kidney": 50, "right_kidney": 60, "none": 0}
description = {}

def get_segdata(json_data, data):
    Z = len(data["segmentation"])
    X = len(data["segmentation"][0])
    Y = len(data["segmentation"][0][0])
    for slice in range(0, Z):
        nbr_drawings = json_data['drawings'][slice][0]['length']
        for draw in range(0, nbr_drawings):
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
            print(points)
            hull = Delaunay(points)
            x, y = np.mgrid[0:X, 0:Y]
            grid = np.vstack([x.ravel(), y.ravel()]).T
            simplex = hull.find_simplex(grid)
            fill = grid[simplex >= 0, :]

            # vlozeni markeru do dat
            dict_description = json_data['drawingsDetails'][slice][0][draw]['longText'].replace("'", '"')
            dict_key = json_data['drawingsDetails'][slice][0][draw]['textExpr']
            dict_value = 100
            print(dict_description)
            if dict_description == '':
                if dict_key == '':
                    print("Drawing is not defined at slice", slice)
                    dict_key = "lbl_" + str(dict_value)
                    data["slab"][dict_key] = dict_value
                else:
                    if dict_key in data["slab"].keys():
                        dict_value = data["slab"][dict_key]
                    else:
                        data["slab"][dict_key] = dict_value
            else:
                dict_description = json.loads(dict_description)
                if "value" in dict_description.keys():
                    dict_value = dict_description["value"]
                elif dict_key in data["slab"].keys():
                    dict_value = data["slab"][dict_key]
                if dict_key != '':
                    data["slab"][dict_key] = dict_value
                else:
                    dict_key = "lbl_" + str(dict_value)
                    data["slab"][dict_key] = dict_value
            for i,j in fill:
                data["segmentation"][slice][i][j] = dict_value
        
            # ziskani zbytku popisu
            if dict_key not in description.keys():
                description[dict_key] = {}
            description[dict_key]["value"] = dict_value

            i_color = draw_info.find('#') + 1
            description[dict_key]["r"] = int(draw_info[i_color:(i_color + 2)], 16)
            description[dict_key]["g"] = int(draw_info[(i_color + 2):(i_color + 4)], 16)
            description[dict_key]["b"] = int(draw_info[(i_color + 4):(i_color + 6)], 16)

            if dict_description != '':
                if "threshold" in dict_description.keys():
                    description[dict_key]["threshold"] = dict_description["threshold"] # nastavit kolem 100 - 120
                if "two" in dict_description.keys():
                    description[dict_key]["two"] = dict_description["two"]
                if "three" in dict_description.keys():
                    description[dict_key]["three"] = dict_description["three"]

def get_seeds(data, label):
    return ((data["segmentation"] != 0).astype('int8') * 2 - 
           (data["segmentation"] == data["slab"][label]).astype('int8'))

def write_to_json(data, data_json, output_name="json_data.json"):
    Z = len(data["segmentation"])
    X = len(data["segmentation"][0])
    Y = len(data["segmentation"][0][0])
    for slice in range(0, Z):
        end = len(data["segmentation"][slice])
        label_array = np.unique(data["segmentation"][slice])[1:end]
        if len(label_array) > 0:
            for lbl in range(0, len(label_array)):
                str_points = ""
                for key, value in data["slab"].items():
                    if value != 0 and value == label_array[lbl]:
                        for x in range(0, X):
                            for y in range(0, Y):
                                if data["segmentation"][slice][x][y] == value:
                                    str_points += ("," if len(str_points) != 0 else "") + str(X - 1 - x) + "," + str(Y - 1 - y)
                        break
                rgba = "rgba(" + str(description[key]["r"])
                rgba += "," + str(description[key]["g"]) + ","
                rgba += str(description[key]["b"]) + ",0.5)"
                data_json["drawings"][slice][0][str(lbl)] = get_str_drawings(key, str_points, rgba, [150, 10 + lbl * 12])
                data_json["drawingsDetails"][slice][0] = [{"id":key, "textExpr":key, "longText":"{\"value\":" + str(value) + "}", "quant":None}]
            data_json["drawings"][slice][0]["length"] = len(label_array)
        else:
            data_json['drawings'][slice][0]["length"] = 0
            data_json['drawingsDetails'][slice][0] = []
    with open(output_name, 'w') as file:
        json.dump(data_json, file)

def get_str_drawings(key, str_points, color, lbl_pos):
    string = "{\"attrs\":{\"name\":\"freeHand-group\",\"visible\":true,\"id\":\"" + key +"\"},"
    string += "\"className\":\"Group\",\"children\":[{\"attrs\":{\"points\":[" + str_points + "],"
    string += "\"stroke\":\"" + color + "\",\"strokeWidth\":2,\"name\":\"shape\",\"tension\":0.5,"
    string += "\"draggable\":true},\"className\":\"Line\"},"
    string += "{\"attrs\":{\"x\":" + str(lbl_pos[0]) + ",\"y\":" + str(lbl_pos[1]) + ",\"name\":\"label\"},"
    string += "\"className\":\"Label\",\"children\":[{\"attrs\":{\"fontSize\":11,\"fontFamily\":\"Verdana\","
    string += "\"fill\":\"" + color + "\",\"name\":\"text\",\"text\":\"" + key + "\"},"
    string += "\"className\":\"Text\"},{\"attrs\":{\"width\":24,\"height\":12},\"className\":\"Tag\"}]}]}"
    return string