import scipy.sparse as sp
import numpy as np
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
            id_2 = [2, 3, 4, 5][id_2 - 12]
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

def frame_to_sparse_matrix(frame):
    triplet_list = frame_to_triplets(frame)
    matrix = triplets_to_matrix(triplet_list)
    sparse_matrix = sp.coo_matrix(matrix)
    return sparse_matrix

def filter_frame(frame):
    filtered_frame = frame.replace(',', '').replace('"', '')
    return filtered_frame

def is_frame_valid(frame):
    # Extract the first list of IDs from the frame
    if "x" in frame:
        return False
    ids_list_1, _ = frame_to_ids(frame)
    
    # look if any value in the first list is greater than 2000
    for value in ids_list_1:
        if value > 2000:
            return False
    return True


def filter_climbs(filtered_df, vgrade=-1, angle=-1):
    climbs_to_remove = []

    if vgrade == -1:
        filtered_df = filtered_df.copy()  # No filtering by vgrade
    elif vgrade == 0:
        filtered_df = filtered_df[filtered_df['vgrade'].isin([0, 1])]
    else:
        vgrade_range = [max(vgrade - 1, 0), min(vgrade + 1, 15)]
        filtered_df = filtered_df[filtered_df['vgrade'].between(vgrade_range[0], vgrade_range[1])]

    if angle != -1:
        filtered_df = filtered_df[filtered_df['angle_y'] == angle]

    valid_mask = filtered_df['frames'].apply(is_frame_valid)
    filtered_df = filtered_df[valid_mask]

    for index, row in filtered_df.iterrows():
        climb_frame = filter_frame(row['frames'])

        remove_climb = False

        words = climb_frame.split('p')[1:]
        for word in words:
            if 1396 <= int(word[:4]) <= 1446:
                remove_climb = True
                break
            elif not word[-2:] in ['12', '13', '14', '15']:
                remove_climb = True
                break
        
        if remove_climb:
            climbs_to_remove.append(index)

    filtered_df.drop(climbs_to_remove, inplace=True)
    
    return filtered_df

def determine_handedness(frame):
    triplet_list = frame_to_triplets(frame)
    print(triplet_list)
    start_positions = [(x, y) for x, y, z in triplet_list if z == 2]
    finish_positions = [(x, y) for x, y, z in triplet_list if z == 4]
    print(start_positions)
    print(finish_positions)

    if not start_positions or not finish_positions:
        return 'unknown'
    
    start_cols = [pos[0] for pos in start_positions]
    finish_cols = [pos[0] for pos in finish_positions]
    print(start_cols)
    print(finish_cols)
    avg_start_col = np.mean(start_cols)
    avg_finish_col = np.mean(finish_cols)

    if avg_start_col < avg_finish_col:
        return 'right'
    else:
        return 'left'

def sort_frame2(frame, handedness=True):
    frame_words = frame.split('p')[1:]

    words_with_rows_cols = []
    for word in frame_words:
        id_1 = int(word.split('r')[0])
        col, row = id_to_coordinate(id_1)
        words_with_rows_cols.append((row, col, word))

    if handedness:
        handedness_type = determine_handedness(frame)
        if handedness_type == 'left':
            # Sort from bottom to top, and then right to left
            sorted_words_with_rows_cols = sorted(words_with_rows_cols, key=lambda x: (x[0], -x[1]))
        else:
            # Sort from bottom to top, and then left to right
            sorted_words_with_rows_cols = sorted(words_with_rows_cols, key=lambda x: (x[0], x[1]))
    else:
        # Treat every climb as right-handed
        sorted_words_with_rows_cols = sorted(words_with_rows_cols, key=lambda x: (x[0], x[1]))

    sorted_frame_words = [word for _, _, word in sorted_words_with_rows_cols]
    sorted_frame = 'p' + 'p'.join(sorted_frame_words)

    return sorted_frame

def sort_frame(frame):
    frame_words = frame.split('p')[1:]

    words_with_rows_cols = []
    for word in frame_words:
        id_1 = int(word.split('r')[0])
        col, row = id_to_coordinate(id_1)
        words_with_rows_cols.append((row, col, word))

    sorted_words_with_rows_cols = sorted(words_with_rows_cols, key=lambda x: (x[0], x[1]))

    sorted_frame_words = [word for _, _, word in sorted_words_with_rows_cols]

    sorted_frame = 'p' + 'p'.join(sorted_frame_words)
    
    return sorted_frame

def filtered_df_to_text_file(filtered_df, file_path='climbs.txt'):
    climb_frames = filtered_df['frames'].apply(filter_frame).apply(lambda x: ' '.join(x.split('p')))

    with open(file_path, 'w') as file:
        for climb_frame in climb_frames:
            file.write(climb_frame + '\n')

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

# def coordinates_to_index2(row_id, col_id):
#     index =  row_id * 35 + col_id
#     if index in id_index_dict: