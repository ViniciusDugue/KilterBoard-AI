import scipy.sparse as sp
import numpy as np
from .embeddings import hold_directions, hold_directions2, hold_magnitudes
import torch
import math
# this is needed because there are circular import issues with process.py and sort_frame.py. 
# Circular imports are not allowed unless if you use importlib but Im lazy
from .process import *



# region sort_frame_1
# def sort_frame(frame):
#     frame_words = frame.split('p')[1:]

#     words_with_rows_cols = []
#     for word in frame_words:
#         id_1 = int(word.split('r')[0])
#         col, row = id_to_coordinate(id_1)
#         words_with_rows_cols.append((row, col, word))

#     sorted_words_with_rows_cols = sorted(words_with_rows_cols, key=lambda x: (x[0], x[1]))

#     sorted_frame_words = [word for _, _, word in sorted_words_with_rows_cols]

#     sorted_frame = 'p' + 'p'.join(sorted_frame_words)
    
#     return sorted_frame

# region sort_frame_2
def sort_frame_2(frame):
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

#region sort_frame_3
def sort_frame_3(frame, row_weight=0.1):
    
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

# region sort_frame_4
def sort_frame_4(frame, row_weight=0.1):

    def weighted_distance(point1, point2, row_weight=0.1):
        x1, y1 = point1
        x2, y2 = point2
        euclidean = math.hypot(x2 - x1, y2 - y1)
        # Lower y-values (higher rows) are better; subtract a weight based on the y-coordinate
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
            col, row, _ = hold[:3]  # Corrected: col is x, row is y
            point = (col, row)
            distance = weighted_distance(center, point, row_weight)
            if distance < min_distance:
                min_distance = distance
                next_hold = hold
        return next_hold

    def find_feet_below_center(center, foot_holds, used_holds):
        center_x, center_y = center
        qualifying_feet = []
        for hold in foot_holds:
            col, row, z, word = hold
            if word in used_holds:
                continue
            # Foot hold must be below the center point's row (higher row number)
            if row >= center_y:
                continue
            qualifying_feet.append(hold)
        return qualifying_feet

    # -------------------- Main Function Logic --------------------

    # Step 1: Split the frame into words
    frame_words = frame.split('p')[1:]  # Split and remove the first empty element

    # Step 2: Get triplet_list from the existing frame_to_triplets function
    triplet_list = frame_to_triplets(frame)  # Assumes (col, row, z)

    # Validate that triplet_list and frame_words are aligned
    if len(triplet_list) != len(frame_words):
        raise ValueError("Mismatch between number of triplets and frame words.")

    # Step 3: Combine triplets with frame words to create (col, row, z, word) tuples
    combined_holds = []
    for triplet, word_part in zip(triplet_list, frame_words):
        col, row, z = triplet  # Corrected: col is x, row is y
        hold_word = f"p{word_part}"  # Reconstruct the original hold string
        combined_hold = (col, row, z, hold_word)  # (col, row, z, word)
        combined_holds.append(combined_hold)

    # Step 4: Categorize holds
    hand_holds = []
    foot_holds = []
    start_holds = []
    finish_holds = []

    for hold in combined_holds:
        col, row, z, word = hold
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

    # Step 5: Sort holds by row ascending (lower to higher), then by column ascending
    sorted_hand_holds = sorted(hand_holds, key=lambda x: (x[1], x[0]))  # Sort by row, then col
    sorted_foot_holds = sorted(foot_holds, key=lambda x: (x[1], x[0]))  # Sort by row, then col

    # Step 6: Initialize sequence and used holds
    sequence = []
    used_holds = set()

    # Initialize pointers
    if len(start_holds) >= 2:
        # Add both start holds in row order
        start_holds_sorted = sorted(start_holds, key=lambda x: x[1])  # Sort by row
        pointer1, pointer2 = start_holds_sorted[0], start_holds_sorted[1]
        sequence.extend([pointer1, pointer2])
        used_holds.update([pointer1[3], pointer2[3]])
    elif len(start_holds) == 1:
        pointer1 = pointer2 = start_holds[0]
        sequence.append(pointer1)
        used_holds.add(pointer1[3])
    else:
        if len(sorted_hand_holds) >= 2:
            pointer1, pointer2 = sorted_hand_holds[0], sorted_hand_holds[1]
            sequence.extend([pointer1, pointer2])
            used_holds.update([pointer1[3], pointer2[3]])
        elif len(sorted_hand_holds) == 1:
            pointer1 = pointer2 = sorted_hand_holds[0]
            sequence.append(pointer1)
            used_holds.add(pointer1[3])
        else:
            pointer1 = pointer2 = None  # No holds to process

    # Step 7: Main loop to build the sequence
    iteration = 1
    while True:
        if not pointer1 or not pointer2:
            break  # No pointers to process

        # Get current hold positions
        pos1 = (pointer1[0], pointer1[1])  # (col, row)
        pos2 = (pointer2[0], pointer2[1])

        # Find center point
        center = find_center(pos1, pos2)

        # Step 7a: Associate foot holds with the center point
        feet_to_add = find_feet_below_center(center, sorted_foot_holds, used_holds)
        if feet_to_add:
            # Sort feet in descending order of row (higher rows first), then ascending col
            feet_sorted = sorted(feet_to_add, key=lambda x: (-x[1], x[0]))
            # Insert feet after the second pointer in the sequence
            index_pointer2 = sequence.index(pointer2)
            for i, foot_hold in enumerate(feet_sorted):
                sequence.insert(index_pointer2 + 1 + i, foot_hold)
                used_holds.add(foot_hold[3])

        # Find next closest hand hold
        next_hold = find_next_closest_hold(center, sorted_hand_holds, used_holds, row_weight)
        if not next_hold:
            break  # No more hand holds to add

        # Add to sequence
        sequence.append(next_hold)
        used_holds.add(next_hold[3])

        # Check if next_hold is a finish hold (even though it's not in hand_holds, precaution)
        if next_hold in finish_holds:
            break  # Reached finish

        # Update pointers
        # Determine which pointer is furthest from next_hold
        next_col, next_row = next_hold[0], next_hold[1]
        distance1 = math.hypot(pos1[0] - next_col, pos1[1] - next_row)
        distance2 = math.hypot(pos2[0] - next_col, pos2[1] - next_row)
        if distance1 > distance2:
            pointer1 = pointer2
            pointer2 = next_hold
        else:
            pointer2 = next_hold

        iteration += 1

    # Step 8: Append any remaining foot holds not yet added
    remaining_foot_holds = [hold for hold in sorted_foot_holds if hold[3] not in used_holds]

    for foot_hold in remaining_foot_holds:
        # Insert foot hold after the first hand hold that is above it
        inserted = False
        for i in range(len(sequence)):
            seq_hold = sequence[i]
            if seq_hold[1] < foot_hold[1]:  # If the current sequence hold's row is above the foot hold's row
                sequence.insert(i + 1, foot_hold)
                used_holds.add(foot_hold[3])
                inserted = True
                break
        if not inserted:
            # If no suitable position found, append at the end
            sequence.append(foot_hold)
            used_holds.add(foot_hold[3])

    # Step 9: Append all finish holds to the end of the sequence to ensure they are last
    for finish_hold in finish_holds:
        if finish_hold[3] not in used_holds:
            sequence.append(finish_hold)
            used_holds.add(finish_hold[3])

    # Step 10: Construct the sorted frame string by joining the original hold strings
    sorted_frame_words = [hold[3] for hold in sequence]
    sorted_frame = ' '.join(sorted_frame_words)  # Joining with space as separator
    return sorted_frame



