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

import math

import math

def sort_frame_5(frame, row_weight=0.1):
    
    def weighted_distance(point1, point2, row_weight=0.1):
        x1, y1 = point1
        x2, y2 = point2
        euclidean = math.hypot(x2 - x1, y2 - y1)
        # Lower y-values are better; subtract a weight based on the y-coordinate
        weighted = euclidean - (row_weight * min(y1, y2))
        return weighted

    def find_center(hold1, hold2):
        x1, y1 = hold1
        x2, y2 = hold2
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def find_next_closest_hold(center, hand_holds, used_holds, row_weight=0.1):
        min_distance = float('inf')
        next_hold = None
        for hold in hand_holds:
            _, _, z, word = hold
            if word in used_holds:
                continue
            # Get coordinates
            row, col, _ = hold[:3]
            # Assuming y is row and x is col
            point = (col, row)
            distance = weighted_distance(center, point, row_weight)
            if distance < min_distance:
                min_distance = distance
                next_hold = hold
        return next_hold

    # Step 1: Split the frame into words
    frame_words = frame.split('p')[1:]  # Split and remove the first empty element

    # Step 2: Get triplet_list from the existing frame_to_triplets function
    triplet_list = frame_to_triplets(frame)  # Assumes (row, col, z)

    # Step 3: Combine triplets with frame words to create (row, col, z, word) tuples
    combined_holds = []
    for triplet, word in zip(triplet_list, frame_words):
        row, col, z = triplet
        hold_word = f"p{word}"  # Reconstruct the original hold string
        combined_hold = (row, col, z, hold_word)  # (row, col, z, word)
        combined_holds.append(combined_hold)

    # Step 4: Categorize holds
    hand_holds = []
    foot_holds = []
    start_holds = []
    finish_holds = []

    for hold in combined_holds:
        row, col, z, word = hold
        if z == 0:
            start_holds.append(hold)
            hand_holds.append(hold)
        elif z == 1:
            hand_holds.append(hold)
        elif z == 2:
            finish_holds.append(hold)
            # Do NOT add finish holds to hand_holds to keep them last
        elif z == 3:
            foot_holds.append(hold)

    # Initialize sequence and used holds
    sequence = []
    used_holds = set()

    # Initialize pointers
    if len(start_holds) >= 2:
        # Add both start holds in row order
        start_holds_sorted = sorted(start_holds, key=lambda x: x[0])  # Sort by row
        pointer1, pointer2 = start_holds_sorted[0], start_holds_sorted[1]
        sequence.extend([pointer1, pointer2])
        used_holds.update([pointer1[3], pointer2[3]])
    elif len(start_holds) == 1:
        pointer1 = pointer2 = start_holds[0]
        sequence.append(pointer1)
        used_holds.add(pointer1[3])

    # Main loop to find the next closest hold
    while True:
        # Get current hold positions
        pos1 = (pointer1[1], pointer1[0])  # (col, row)
        pos2 = (pointer2[1], pointer2[0])

        # Find center point
        center = find_center(pos1, pos2)

        # Find next closest hand hold
        next_hold = find_next_closest_hold(center, hand_holds, used_holds, row_weight)
        if not next_hold:
            break

        # Add to sequence
        sequence.append(next_hold)
        used_holds.add(next_hold[3])

        # Check if next_hold is a finish hold (even though it's not in hand_holds, precaution)
        if next_hold in finish_holds:
            break  # Reached finish

        # Update pointers
        # Determine which pointer is furthest from next_hold
        distance1 = math.hypot(pos1[0] - next_hold[1], pos1[1] - next_hold[0])
        distance2 = math.hypot(pos2[0] - next_hold[1], pos2[1] - next_hold[0])
        if distance1 > distance2:
            pointer1 = next_hold
        else:
            pointer2 = next_hold

    # Add any missing hand holds to the sequence
    all_hand_holds = set(hold[3] for hold in hand_holds)
    missing_holds = all_hand_holds - used_holds
    for hold_word in missing_holds:
        # Find the hold details
        hold = next((h for h in hand_holds if h[3] == hold_word), None)
        if hold:
            # Find the closest hold in the current sequence
            min_distance = float('inf')
            closest_hold_in_seq = None
            hold_pos = (hold[1], hold[0])
            for seq_hold in sequence:
                seq_pos = (seq_hold[1], seq_hold[0])
                distance = math.hypot(hold_pos[0] - seq_pos[0], hold_pos[1] - seq_pos[1])
                if distance < min_distance:
                    min_distance = distance
                    closest_hold_in_seq = seq_hold
            # Insert the missing hold near the closest hold in the sequence
            if closest_hold_in_seq:
                index = sequence.index(closest_hold_in_seq)
                sequence.insert(index + 1, hold)
                used_holds.add(hold[3])

    # Append all finish holds to the end of the sequence to ensure they are last
    for finish_hold in finish_holds:
        if finish_hold[3] not in used_holds:
            sequence.append(finish_hold)
            used_holds.add(finish_hold[3])

    # Add all footholds to the end of the sequence
    for foot_hold in foot_holds:
        if foot_hold[3] not in used_holds:
            sequence.append(foot_hold)
            used_holds.add(foot_hold[3])

    # Construct the sorted frame string
    sorted_frame_words = [hold[3] for hold in sequence]  # Using hold[3] which is 'p<row>r<z>'
    sorted_frame = ' '.join(sorted_frame_words)  # Joining with space as separator

    return sorted_frame






def sort_frame_4(frame, row_weight=0.1, max_distance_below=6.5):
    # Step 1: Split the frame into words
    frame_words = frame.split('p')[1:]  # Split and remove the first empty element

    # Step 2: Get triplet_list from the existing frame_to_triplets function
    triplet_list = frame_to_triplets(frame)  # Assumes (row, col, z)

    combined_holds = []
    for triplet, word in zip(triplet_list, frame_words):
        row, col, z = triplet
        hold_word = f"p{word}"  # Reconstruct the original hold string
        combined_hold = (row, col, z, hold_word)  # (row, col, z, word)
        combined_holds.append(combined_hold)

    # Step 4: Categorize holds
    hand_holds = []
    foot_holds = []
    start_holds = []
    finish_holds = []
    for hold in combined_holds:
        row, col, z, word = hold
        if z == 0:
            start_holds.append(hold)
            hand_holds.append(hold)
        elif z == 1:
            hand_holds.append(hold)
        elif z == 2:
            finish_holds.append(hold)
            hand_holds.append(hold)
        elif z == 3:
            foot_holds.append(hold)

    # Step 5: Sort holds by row ascending, then by column ascending
    sorted_hand_holds = sorted(hand_holds, key=lambda x: (x[0], x[1]))
    sorted_foot_holds = sorted(foot_holds, key=lambda x: (x[0], x[1]))

    # Step 6: Initialize new_sequence and used_holds
    new_sequence = []
    used_holds = set()

    def find_closest_hold(current, candidates, weight=0.1):
        curr_row, curr_col, _, _ = current
        closest_hold = None
        closest_modified_dist = float('inf')

        for hold in candidates:
            if tuple(hold) in used_holds:
                continue
            hold_row, hold_col, z, word = hold
            if hold_row < curr_row:
                continue  # Only consider holds at or above the current row

            distance = np.sqrt((hold_row - curr_row) ** 2 + (hold_col - curr_col) ** 2)
            modified_dist = distance + weight * (hold_row - curr_row)

            if modified_dist < closest_modified_dist:
                closest_modified_dist = modified_dist
                closest_hold = hold

        return closest_hold
    
    # Initialize with start holds or first two hand holds
    if len(start_holds) >= 2:
        start_holds_sorted = sorted(start_holds, key=lambda x: x[0])
        pointer1, pointer2 = start_holds_sorted[:2]
        new_sequence.extend([pointer1, pointer2])
        used_holds.update([tuple(pointer1), tuple(pointer2)])
    elif len(start_holds) == 1:
        pointer1 = pointer2 = start_holds[0]
        new_sequence.append(pointer1)
        used_holds.add(tuple(pointer1))
    else:
        if len(sorted_hand_holds) >= 2:
            pointer1, pointer2 = sorted_hand_holds[:2]
            new_sequence.extend([pointer1, pointer2])
            used_holds.update([tuple(pointer1), tuple(pointer2)])
        elif len(sorted_hand_holds) == 1:
            pointer1 = pointer2 = sorted_hand_holds[0]
            new_sequence.append(pointer1)
            used_holds.add(tuple(pointer1))
        else:
            pointer1 = pointer2 = None

    # Helper function to find the closest hold
    def find_closest_hold(current, candidates, weight=0.1):
        curr_row, curr_col, _, _ = current
        closest_hold = None
        closest_modified_dist = float('inf')

        for hold in candidates:
            if tuple(hold) in used_holds:
                continue
            hold_row, hold_col, z, word = hold
            if hold_row < curr_row:
                continue  # Only consider holds at or above the current row

            distance = np.sqrt((hold_row - curr_row) ** 2 + (hold_col - curr_col) ** 2)
            modified_dist = distance + weight * (hold_row - curr_row)

            if modified_dist < closest_modified_dist:
                closest_modified_dist = modified_dist
                closest_hold = hold

        return closest_hold

    # Step 7: Build the new_sequence
    while len(new_sequence) < len(sorted_hand_holds) + len(sorted_foot_holds):
        if not new_sequence:
            break
        curr_hold = new_sequence[-1]
        closest_hold = find_closest_hold(curr_hold, sorted_hand_holds, row_weight)

        if closest_hold:
            # Add holds below or at the current row within max_distance_below
            holds_within = [
                hold for hold in sorted_hand_holds
                if hold[0] <= curr_hold[0]
                   and tuple(hold) not in used_holds
                   and hold != closest_hold
                   and np.sqrt((hold[0] - curr_hold[0]) ** 2 + (hold[1] - curr_hold[1]) ** 2) <= max_distance_below
            ]
            holds_within_sorted = sorted(
                holds_within,
                key=lambda hold: np.sqrt((hold[0] - curr_hold[0]) ** 2 + (hold[1] - curr_hold[1]) ** 2)
            )
            for hold in holds_within_sorted:
                new_sequence.append(hold)
                used_holds.add(tuple(hold))

            # Add holds below or at the current row beyond max_distance_below
            holds_beyond = [
                hold for hold in sorted_hand_holds
                if hold[0] <= curr_hold[0]
                   and tuple(hold) not in used_holds
                   and hold != closest_hold
                   and np.sqrt((hold[0] - curr_hold[0]) ** 2 + (hold[1] - curr_hold[1]) ** 2) > max_distance_below
            ]
            holds_beyond_sorted = sorted(
                holds_beyond,
                key=lambda hold: np.sqrt((hold[0] - curr_hold[0]) ** 2 + (hold[1] - curr_hold[1]) ** 2)
            )
            for hold in holds_beyond_sorted:
                new_sequence.append(hold)
                used_holds.add(tuple(hold))

            # Add the closest hold
            new_sequence.append(closest_hold)
            used_holds.add(tuple(closest_hold))

            # Update pointers
            pointer1, pointer2 = pointer2, closest_hold
        else:
            # Add remaining holds within distance
            remaining_within = [
                hold for hold in sorted_hand_holds
                if tuple(hold) not in used_holds
                   and hold[0] <= curr_hold[0]
                   and np.sqrt((hold[0] - curr_hold[0]) ** 2 + (hold[1] - curr_hold[1]) ** 2) <= max_distance_below
            ]
            if remaining_within:
                remaining_within_sorted = sorted(
                    remaining_within,
                    key=lambda hold: np.sqrt((hold[0] - curr_hold[0]) ** 2 + (hold[1] - curr_hold[1]) ** 2)
                )
                for hold in remaining_within_sorted:
                    new_sequence.append(hold)
                    used_holds.add(tuple(hold))
            else:
                # Add any remaining holds below or at the current row
                remaining_beyond = [
                    hold for hold in sorted_hand_holds
                    if tuple(hold) not in used_holds
                       and hold[0] <= curr_hold[0]
                ]
                if not remaining_beyond:
                    # Add any remaining holds
                    remaining = [hold for hold in sorted_hand_holds if tuple(hold) not in used_holds]
                    if not remaining:
                        break
                    remaining_sorted = sorted(
                        remaining,
                        key=lambda hold: np.sqrt((hold[0] - curr_hold[0]) ** 2 + (hold[1] - curr_hold[1]) ** 2)
                    )
                    for hold in remaining_sorted:
                        new_sequence.append(hold)
                        used_holds.add(tuple(hold))
                else:
                    remaining_beyond_sorted = sorted(
                        remaining_beyond,
                        key=lambda hold: np.sqrt((hold[0] - curr_hold[0]) ** 2 + (hold[1] - curr_hold[1]) ** 2)
                    )
                    for hold in remaining_beyond_sorted:
                        new_sequence.append(hold)
                        used_holds.add(tuple(hold))

    # Step 8: Add footholds at the end
    for foot_hold in sorted_foot_holds:
        if tuple(foot_hold) not in used_holds:
            new_sequence.append(foot_hold)
            used_holds.add(tuple(foot_hold))

    # Step 9: Remove duplicates while preserving order
    unique_sequence = []
    seen = set()
    for hold in new_sequence:
        if tuple(hold) not in seen:
            unique_sequence.append(hold)
            seen.add(tuple(hold))

    # Step 10: Reconstruct the sorted frame string by joining the original hold strings
    sorted_frame_words = [hold[3] for hold in unique_sequence]
    sorted_frame = ' '.join(sorted_frame_words)

    return sorted_frame


def sort_frame_3(frame):
    # Step 1: Parse the frame into (row, col, word) tuples
    frame_words = frame.split('p')[1:]  # Split and remove the first empty element

    words_with_rows_cols = []
    for word in frame_words:
        # Extract the numeric part before 'r'
        hold_id_str = ''.join(filter(str.isdigit, word.split('r')[0]))
        if hold_id_str == '':
            raise ValueError(f"Invalid hold format: {word}")
        hold_id = int(hold_id_str)
        col, row = id_to_coordinate(hold_id)
        # Ensure row and col are within 0-34 to prevent wrap-around
        if not (0 <= row < 35) or not (0 <= col < 35):
            raise ValueError(f"Hold ID {hold_id} maps to invalid coordinates: ({col}, {row})")
        words_with_rows_cols.append((row, col, word))

    # Step 2: Sort the holds initially by row (ascending) and then by column (ascending)
    sorted_words_with_rows_cols = sorted(words_with_rows_cols, key=lambda x: (x[0], x[1]))

    # Initialize sequence and used holds
    new_sequence = []
    used_holds = set()

    if not sorted_words_with_rows_cols:
        return frame  # Return original frame if no holds are present

    # Start with the first hold as the current hold
    curr_hold = sorted_words_with_rows_cols[0]
    new_sequence.append(curr_hold)
    used_holds.add(curr_hold)

    def find_closest_hold(current, candidates, weight= 0.1):
        curr_row, curr_col, _ = current
        closest_hold = None
        closest_modified_dist = float('inf')

        for hold in candidates:
            hold_row, hold_col, _ = hold
            if hold in used_holds:
                continue
            if hold_row < curr_row:
                continue  # Only consider holds at or above the current row

            # Calculate Euclidean distance
            distance = np.sqrt((hold_row - curr_row) ** 2 + (hold_col - curr_col) ** 2)
            # Modify distance to prioritize lower row holds
            modified_dist = distance + weight * (hold_row - curr_row)

            if modified_dist < closest_modified_dist:
                closest_modified_dist = modified_dist
                closest_hold = hold

        return closest_hold
    max_distance_below = 6.5

    while len(new_sequence) < len(sorted_words_with_rows_cols):
        curr_hold = new_sequence[-1]
        curr_row, curr_col, _ = curr_hold

        # Step 3: Find the closest hold at or above the current hold's row
        closest_hold = find_closest_hold(curr_hold, sorted_words_with_rows_cols)

        if closest_hold:
            # Step 4a: Add holds below or at the current row within max_distance_below
            holds_below_or_at_within_distance = [
                hold for hold in sorted_words_with_rows_cols
                if hold[0] <= curr_row
                   and hold not in used_holds
                   and hold != closest_hold
                   and np.sqrt((hold[0] - curr_row) ** 2 + (hold[1] - curr_col) ** 2) <= max_distance_below
            ]

            # Sort holds within distance by distance ascending
            holds_below_or_at_within_distance_sorted = sorted(
                holds_below_or_at_within_distance,
                key=lambda hold: np.sqrt((hold[0] - curr_row) ** 2 + (hold[1] - curr_col) ** 2)
            )

            for hold in holds_below_or_at_within_distance_sorted:
                new_sequence.append(hold)
                used_holds.add(hold)

            # Step 4b: Add holds below or at the current row beyond max_distance_below
            holds_below_or_at_beyond_distance = [
                hold for hold in sorted_words_with_rows_cols
                if hold[0] <= curr_row
                   and hold not in used_holds
                   and hold != closest_hold
                   and np.sqrt((hold[0] - curr_row) ** 2 + (hold[1] - curr_col) ** 2) > max_distance_below
            ]

            # Sort holds beyond distance by distance ascending
            holds_below_or_at_beyond_distance_sorted = sorted(
                holds_below_or_at_beyond_distance,
                key=lambda hold: np.sqrt((hold[0] - curr_row) ** 2 + (hold[1] - curr_col) ** 2)
            )

            for hold in holds_below_or_at_beyond_distance_sorted:
                new_sequence.append(hold)
                used_holds.add(hold)

            # Now add the closest hold
            new_sequence.append(closest_hold)
            used_holds.add(closest_hold)

            # Update current hold to the closest hold
            curr_hold = closest_hold
        else:
            # If no closest hold is found, add any remaining holds below or at the current row
            # that are within max_distance_below
            remaining_holds_within_distance = [
                hold for hold in sorted_words_with_rows_cols
                if hold not in used_holds
                   and hold[0] <= curr_row
                   and np.sqrt((hold[0] - curr_row) ** 2 + (hold[1] - curr_col) ** 2) <= max_distance_below
            ]

            if remaining_holds_within_distance:
                # Sort by distance ascending
                remaining_holds_within_distance_sorted = sorted(
                    remaining_holds_within_distance,
                    key=lambda hold: np.sqrt((hold[0] - curr_row) ** 2 + (hold[1] - curr_col) ** 2)
                )
                for hold in remaining_holds_within_distance_sorted:
                    new_sequence.append(hold)
                    used_holds.add(hold)
            else:
                # If no holds within distance, add any remaining holds below or at the current row
                remaining_holds_beyond_distance = [
                    hold for hold in sorted_words_with_rows_cols
                    if hold not in used_holds
                       and hold[0] <= curr_row
                ]

                if not remaining_holds_beyond_distance:
                    # If no holds are left below or at the current row, search for any remaining holds
                    remaining_holds = [
                        hold for hold in sorted_words_with_rows_cols
                        if hold not in used_holds
                    ]
                    if not remaining_holds:
                        break  # All holds have been added

                    # Sort remaining holds by distance ascending
                    remaining_holds_sorted = sorted(
                        remaining_holds,
                        key=lambda hold: np.sqrt((hold[0] - curr_row) ** 2 + (hold[1] - curr_col) ** 2)
                    )

                    for hold in remaining_holds_sorted:
                        new_sequence.append(hold)
                        used_holds.add(hold)
                else:
                    # Sort holds beyond distance by distance ascending
                    remaining_holds_beyond_distance_sorted = sorted(
                        remaining_holds_beyond_distance,
                        key=lambda hold: np.sqrt((hold[0] - curr_row) ** 2 + (hold[1] - curr_col) ** 2)
                    )
                    for hold in remaining_holds_beyond_distance_sorted:
                        new_sequence.append(hold)
                        used_holds.add(hold)

    # Ensure all holds are unique and preserve the order
    unique_sequence = []
    seen = set()
    for hold in new_sequence:
        if hold not in seen:
            unique_sequence.append(hold)
            seen.add(hold)

    # Construct the sorted frame string
    sorted_frame_words = [hold[2] for hold in unique_sequence]
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