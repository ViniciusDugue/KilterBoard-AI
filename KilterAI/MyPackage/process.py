import scipy.sparse as sp
import numpy as np
from .embeddings import hold_directions, hold_directions2, hold_magnitudes
import torch
import pandas as pd
# region frame

def map_vgrade(difficulty):
    ranges = [
        (10, 12, 0), (12, 14, 1), (14, 16, 2), (16, 18, 3), 
        (18, 20, 4), (20, 22, 5), (22, 23, 6), (23, 24, 7), 
        (24, 26, 8), (26, 28, 9), (28, 29, 10), (29, 30, 11), 
        (30, 31, 12), (31, 32, 13), (32, 33, 14), (33, float('inf'), 15)
    ]
    for lower, upper, grade in ranges:
        if lower <= difficulty < upper:
            return grade

    return None

# print(id_to_coordinate(1476))
# print(id_to_coordinate(1539))

def frame_to_ids(frame):
    ids_list_1 = []
    ids_list_2 = []
    
    # Split the frame string based on the 'p' delimiter
    filtered_frame = frame.replace(',', '').replace('"', '')
    frame_parts = filtered_frame.split('p')[1:]
    
    for entry in frame_parts: 
        parts = entry.split('r')
        id_1 = int(parts[0])
        id_2 = int(parts[1])

        ids_list_1.append(id_1)
        if id_2 in [12, 13, 14, 15]:
            id_2 = [0, 1, 2, 3][id_2 - 12]
        ids_list_2.append(id_2)
    return ids_list_1, ids_list_2 

# takes in frame and converts to list of triplets with xy being coordiante in board matrix and z being the value
def frame_to_triplets(frame):
    ids_list_1, ids_list_2 = frame_to_ids(frame)
    
    triplet_list = []
    for id_1, id_2 in zip(ids_list_1, ids_list_2):
        coordinate = id_to_coordinate(id_1)
        triplet = (coordinate[0],coordinate[1], id_2)
        triplet_list.append(triplet)
    
    return triplet_list

# frame = "p1127r12p1164r12p1233r13p1235r13p1283r13p1287r13p1299r13p1348r13p1379r14p1458r15p1507r15"
# triplet_list = frame_to_triplets(frame)
# print("Triplet List:", triplet_list)

def triplets_to_matrix(triplet_list):
    # Initialize a 35x35 matrix with zeros
    matrix = [[0] * 35 for _ in range(35)]

    for triplet in triplet_list:
        x, y, z = triplet
        matrix[y][x] = z
    
    return matrix

def process_frame(frame):
    filtered_frame = frame.replace(',', '').replace('"', '')
    return filtered_frame

def is_frame_valid(frame):
    """Checks if a frame is valid based on multiple criteria."""
    if "x" in frame:
        return False 

    ids_list_1, _ = frame_to_ids(frame)
    
    if any(value > 2000 for value in ids_list_1):
        return False

    climb_frame = process_frame(frame)
    words = climb_frame.split('p')[1:]

    if any(1396 <= int(word[:4]) <= 1446 or word[-2:] not in ['12', '13', '14', '15'] for word in words):
        return False 

    return True

def hold_id_and_val_to_frame(hold_ids, vals):
    frame = ' '.join([f"p{hold_id}r{val + 12}" for hold_id, val in zip(hold_ids, vals)])
    return frame

# region subframe
# id - original hold id from dataset (1074 - 1396 & 1448 - 1599)
# index- index of the climb based on its position in the matrix 35x35 (0-1225)
# index2(rename to hold_id)- index of the climb based on its row col values (0 - 475)( better version of class_id
# class_id -this one is bad. The top holds on board end at 325ish whereas the 400s holds are foot holds at the bottom of the board (0-475)
# hold_val_class_id(rename to holdval_id)- combines class_id and val_id (0 - 1904)

def id_to_index(id):
    
    if id <=1089: #bottom large (row) 17x1
        index_offset = 35
        row_index = 16 - (id - 1074)
        final_index = index_offset + 2 * row_index
    elif id <= 1395: #big holds (matrix) 17x18
        index_offset = 35 + 35
        index = id-1089
        row = index//17
        row_index = index%17
        final_index = index_offset + row * 70 + 2 *row_index   
    elif id <=1464: #bottom small (row) 18x1
        row_index = 17 - (id - 1447)
        final_index = 2 * row_index + 1
    elif id <= 1599: # small holds (matrix) 9x15
        index_offset = 35 + 35 + 35 + 1
        index = id-1464
        row = index//9
        row_index = index% 9
        final_index = index_offset + row * 70 + 4 * (row_index-1)
    else:
        final_index = id
        print(id) 
    return final_index

# create dictionaries for converting from id to index and id to index2
id_index_dict = {}
for id in range(1073, 1396):
    id_index_dict[id] = id_to_index(id)
for id in range(1447, 1600):
    id_index_dict[id] = id_to_index(id)

sorted_id_index_tuples = sorted(id_index_dict.items(), key=lambda item: item[1])

index2_to_id_dict = {index: id for index, (id, _) in enumerate(sorted_id_index_tuples)}
id_to_index2_dict = {id: index for index, id in index2_to_id_dict.items()}

def id_to_index2(id):
    return id_to_index2_dict.get(id, None)

def index2_to_id(index2):
    return index2_to_id_dict.get(index2, None)

def id_to_coordinate(id):
    index = id_to_index(id)-1
    # print(f"id: {id} index: {index}")
    x = (index % 35)
    y = index//35
    if y >=35:
        y-=2 #because the top 2 rows of large hand holds dont have any feet so its offset
    if y ==35:
        y-=1 # idk I guess this is for the top right hold on the board??
    if y <= 31 and y >=2 and (y -1) %4 == 0 and x!=34:
        x +=2 # because each alternating row of feet are offset by 2 on the x axis
    
    return (x,y)

def id_to_class_id(id):
    if id <=1089: # 0-16
        class_id = id - 1073 
    elif id <=1395: # 17-322
        class_id = id - 1090 + 17
    elif id <= 1464: # 323-340
        class_id = id - 1447 + 323
    elif id <= 1599: # 341-475
        class_id = id - 1465 + 341
    return class_id

def class_id_to_id(class_id):
    if class_id <= 16:  # 0-16
        id = class_id + 1073
    elif class_id <= 322:  # 17-322
        id = class_id + 1090 - 17
    elif class_id <= 340:  # 323-340
        id = class_id + 1447 - 323
    elif class_id <= 475:  # 341-475
        id = class_id + 1465 - 341 
    return id

def ids_to_hold_val_class_id(hold_class_id, val_class_id):
    hold_val_class_id = (hold_class_id) * 4 + val_class_id
    return hold_val_class_id

def hold_val_class_id_to_ids(hold_val_class_id):
    val_class_id = hold_val_class_id % 4
    hold_class_id = int((hold_val_class_id - val_class_id) /4)
    return hold_class_id, val_class_id

# region hold vectors
def get_hold_vector(row, col):
    if not isinstance(row, int):
        raise TypeError(f"Row index must be an integer, got {type(row)} row: {row}")
    if not isinstance(col, int):
        raise TypeError(f"Column index must be an integer, got {type(col)} col: {col}")
    
    if row > 34 or col > 34:
        if row > col:
            raise ValueError(f"Row index {row} exceeds maximum allowed value of 34.")
        else:
            raise ValueError(f"Column index {col} exceeds maximum allowed value of 34.")
    
    if max(hold_directions[34 - row][col]) > 1:
        index = int(hold_directions[34 - row][col][0]) - 1
        hold_direction = hold_directions2[index].tolist()
    else:
        hold_direction = hold_directions[34 - row][col].tolist()
    
    hold_magnitude = hold_magnitudes[34 - row][col][0]
    
    return hold_direction, hold_magnitude

def unit_vector_to_sin(vector):
    x, y = vector
    angle_radians = np.arctan2(y, x)
    sin_value = np.sin(angle_radians)
    return sin_value

def unit_vector_to_cos(vector):
    x, y = vector
    angle_radians = np.arctan2(y, x)
    cos_value = np.cos(angle_radians)
    return cos_value

def modified_hold_quality(climb_angle, hold_quality):
    if isinstance(climb_angle, torch.Tensor):
        climb_angle_cpu = climb_angle.cpu().numpy()
    else:
        climb_angle_cpu = np.array(climb_angle)
    climb_angle_radians = np.deg2rad(climb_angle_cpu)
    modified_quality = np.cos(climb_angle_radians * hold_quality)
    return modified_quality

def interhold_angle(a, b):
    x1, y1 = a
    x2, y2 = b
    angle_radians = np.arctan2(y2 - y1, x2 - x1)
    return angle_radians

def average_unit_vectors(vectors):
    if len(vectors) == 1:
        return vectors[0]
    elif len(vectors) == 2:
        avg_x = (vectors[0][0] + vectors[1][0]) / 2
        avg_y = (vectors[0][1] + vectors[1][1]) / 2
        norm = np.sqrt(avg_x**2 + avg_y**2)
        return [avg_x / norm, avg_y / norm]
    else:
        raise ValueError("The input should be a list containing one or two unit vectors.")

# region other
def coordinates_distribution_to_index2(row_id, col_pred):
    for col_id in col_pred:
        index = row_id * 35 + col_id
        print(index)
        if index in id_index_dict:
            return id_to_index2(id_index_dict[index])

    print("Invalid")
    for col_id in col_pred:
        if col_id + 1 < 35:
            index_right = row_id * 35 + (col_id + 1)
            if index_right in id_index_dict:
                print(index_right)
                return id_to_index2(id_index_dict[index_right])
        
        if col_id - 1 >= 0:
            index_left = row_id * 35 + (col_id - 1)
            if index_left in id_index_dict:
                print(index_left)
                return id_to_index2(id_index_dict[index_left])
        if col_id + 2 < 35:
            index_right = row_id * 35 + (col_id + 2)
            if index_right in id_index_dict:
                print(index_right)
                return id_to_index2(id_index_dict[index_right])
        
        if col_id - 2 >= 0:
            index_left = row_id * 35 + (col_id - 2)
            if index_left in id_index_dict:
                print(index_left)
                return id_to_index2(id_index_dict[index_left])
    return None
