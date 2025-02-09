from .process import *
import math

# region ids to coords test
def ids_to_coordinates():
    # Iterate through the first range: 1073-1395
    for id in range(1073, 1396):
        coord = id_to_coordinate(id)
        # print(f"ID {id} converted to coordinates: {coord}")
    
    # Iterate through the second range: 1447-1599
    for id in range(1447, 1600):
        coord = id_to_coordinate(id)
        print(f"ID {id} converted to coordinates: {coord}")

# region benchmark 1
def benchmark1(frame, row_weight=0.1):
    def weighted_distance(point1, point2, row_weight=0.1):
        x1, y1 = point1
        x2, y2 = point2
        euclidean = math.hypot(x2 - x1, y2 - y1)
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
            col, row, _ = hold[:3]
            point = (col, row)
            distance = weighted_distance(center, point, row_weight)
            if distance < min_distance:
                min_distance = distance
                next_hold = hold
        return next_hold

    triplet_list = frame_to_triplets(frame)
    frame_words = frame.split('p')[1:]
    combined_holds = []
    for triplet, word_part in zip(triplet_list, frame_words):
        col, row, z = triplet
        hold_word = f"p{word_part}"
        combined_hold = (col, row, z, hold_word)
        combined_holds.append(combined_hold)

    hand_holds = [hold for hold in combined_holds if hold[2] in (0, 1)]
    sorted_hand_holds = sorted(hand_holds, key=lambda x: (x[1], x[0]))
    start_holds = [hold for hold in sorted_hand_holds if hold[2] == 0]

    sequence = []
    used_holds = set()
    total_distance = 0.0

    if len(start_holds) >= 2:
        start_holds_sorted = sorted(start_holds, key=lambda x: x[1])
        pointer1, pointer2 = start_holds_sorted[0], start_holds_sorted[1]
        sequence.extend([pointer1, pointer2])
        used_holds.update([pointer1[3], pointer2[3]])
        center = find_center(pointer1[:2], pointer2[:2])
    elif len(start_holds) == 1:
        pointer1 = pointer2 = start_holds[0]
        sequence.append(pointer1)
        used_holds.add(pointer1[3])
        center = pointer1[:2]
    else:
        if len(sorted_hand_holds) >= 2:
            pointer1, pointer2 = sorted_hand_holds[0], sorted_hand_holds[1]
            sequence.extend([pointer1, pointer2])
            used_holds.update([pointer1[3], pointer2[3]])
            center = find_center(pointer1[:2], pointer2[:2])
        elif len(sorted_hand_holds) == 1:
            pointer1 = pointer2 = sorted_hand_holds[0]
            sequence.append(pointer1)
            used_holds.add(pointer1[3])
            center = pointer1[:2]
        else:
            return [], 0.0

    while True:
        next_hold = find_next_closest_hold(center, sorted_hand_holds, used_holds, row_weight)
        if not next_hold:
            break
        next_center = find_center(center, next_hold[:2])
        distance = math.hypot(next_center[0] - center[0], next_center[1] - center[1])
        total_distance += distance
        used_holds.add(next_hold[3])
        center = next_center

    return total_distance

#region benchmark 2
def benchmark2(frame, row_weight=0.1, percent=50.0):
    def weighted_distance(point1, point2, row_weight=0.1):
        x1, y1 = point1
        x2, y2 = point2
        euclidean = math.hypot(x2 - x1, y2 - y1)
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
            col, row, _ = hold[:3]
            point = (col, row)
            distance = weighted_distance(center, point, row_weight)
            if distance < min_distance:
                min_distance = distance
                next_hold = hold
        return next_hold

    triplet_list = frame_to_triplets(frame)
    frame_words = frame.split('p')[1:]
    combined_holds = []
    for triplet, word_part in zip(triplet_list, frame_words):
        col, row, z = triplet
        hold_word = f"p{word_part}"
        combined_hold = (col, row, z, hold_word)
        combined_holds.append(combined_hold)

    hand_holds = [hold for hold in combined_holds if hold[2] in (0, 1)]
    sorted_hand_holds = sorted(hand_holds, key=lambda x: (x[1], x[0]))
    start_holds = [hold for hold in sorted_hand_holds if hold[2] == 0]

    sequence = []
    used_holds = set()
    total_distance = 0.0
    count = 0
    distances = []

    if len(start_holds) >= 2:
        start_holds_sorted = sorted(start_holds, key=lambda x: x[1])
        pointer1, pointer2 = start_holds_sorted[0], start_holds_sorted[1]
        sequence.extend([pointer1, pointer2])
        used_holds.update([pointer1[3], pointer2[3]])
        center = find_center(pointer1[:2], pointer2[:2])
    elif len(start_holds) == 1:
        pointer1 = pointer2 = start_holds[0]
        sequence.append(pointer1)
        used_holds.add(pointer1[3])
        center = pointer1[:2]
    else:
        if len(sorted_hand_holds) >= 2:
            pointer1, pointer2 = sorted_hand_holds[0], sorted_hand_holds[1]
            sequence.extend([pointer1, pointer2])
            used_holds.update([pointer1[3], pointer2[3]])
            center = find_center(pointer1[:2], pointer2[:2])
        elif len(sorted_hand_holds) == 1:
            pointer1 = pointer2 = sorted_hand_holds[0]
            sequence.append(pointer1)
            used_holds.add(pointer1[3])
            center = pointer1[:2]
        else:
            return 0.0, 0.0, 0.0, 0.0

    while True:
        next_hold = find_next_closest_hold(center, sorted_hand_holds, used_holds, row_weight)
        if not next_hold:
            break
        next_center = find_center(center, next_hold[:2])
        distance = math.hypot(next_center[0] - center[0], next_center[1] - center[1])
        total_distance += distance
        distances.append(distance)
        count += 1
        used_holds.add(next_hold[3])
        center = next_center

    if count > 0:
        average_distance = total_distance / count
        k = max(1, int(math.ceil(len(distances) * (percent / 100.0))))
        top_distances = sorted(distances, reverse=True)[:k]
        top_n_distance_sum = sum(top_distances)
        top_n_average_distance = top_n_distance_sum / k if k > 0 else 0.0
        longest_distance = max(distances)
    else:
        average_distance = 0.0
        top_n_distance_sum = 0.0
        top_n_average_distance = 0.0
        longest_distance = 0.0

    return total_distance, average_distance, top_n_distance_sum, top_n_average_distance, longest_distance

# region benchmark 3
def benchmark3(frame, row_weight=0.1, percent=50.0, min_product=0.1):
    """
    Calculates weighted distances between consecutive center points, incorporating hold magnitudes,
    and returns comprehensive benchmarking metrics.

    Parameters:
        frame (str): The frame string representing the sequence of holds.
        row_weight (float): Weight factor to adjust the influence of row positions.
        percent (float): Percentage to determine top N weighted distances for additional metrics.
        min_product (float): Minimum threshold for the product of hold magnitudes to prevent division by zero.

    Returns:
        tuple: (
            total_weighted_distance (float),
            average_weighted_distance (float),
            top_n_weighted_distance_sum (float),
            top_n_average_weighted_distance (float),
            highest_weighted_distance (float)
        )
    """
    def weighted_distance(point1, point2, row_weight=0.1):
        x1, y1 = point1
        x2, y2 = point2
        euclidean = math.hypot(x2 - x1, y2 - y1)
        # Adjust distance based on row_weight
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
            col, row, _ = hold[:3]
            point = (col, row)
            distance = weighted_distance(center, point, row_weight)
            if distance < min_distance:
                min_distance = distance
                next_hold = hold
        return next_hold

    # Step 1: Parse the frame into triplets
    triplet_list = frame_to_triplets(frame)  # List of (x, y, z)
    frame_words = frame.split('p')[1:]       # List of 'id1rval1', 'id2rval2', ...

    # Step 2: Combine triplets with frame words to create (col, row, z, word) tuples
    combined_holds = []
    for triplet, word in zip(triplet_list, frame_words):
        x, y, z = triplet
        hold_word = f"p{word}"  # Reconstruct the original hold string
        combined_hold = (x, y, z, hold_word)  # (col, row, z, word)
        combined_holds.append(combined_hold)

    # Step 3: Categorize holds
    hand_holds = [hold for hold in combined_holds if hold[2] in (0, 1)]
    sorted_hand_holds = sorted(hand_holds, key=lambda x: (x[1], x[0]))  # Sort by row, then col
    start_holds = [hold for hold in sorted_hand_holds if hold[2] == 0]

    sequence = []
    used_holds = set()
    total_weighted_distance = 0.0
    weighted_distances = []

    # Initialize pointers
    if len(start_holds) >= 2:
        # Add both start holds in row order
        start_holds_sorted = sorted(start_holds, key=lambda x: x[1])  # Sort by row
        pointer1, pointer2 = start_holds_sorted[0], start_holds_sorted[1]
        sequence.extend([pointer1, pointer2])
        used_holds.update([pointer1[3], pointer2[3]])
        center = find_center((pointer1[0], pointer1[1]), (pointer2[0], pointer2[1]))
    elif len(start_holds) == 1:
        pointer1 = pointer2 = start_holds[0]
        sequence.append(pointer1)
        used_holds.add(pointer1[3])
        center = (pointer1[0], pointer1[1])
    else:
        if len(sorted_hand_holds) >= 2:
            pointer1, pointer2 = sorted_hand_holds[0], sorted_hand_holds[1]
            sequence.extend([pointer1, pointer2])
            used_holds.update([pointer1[3], pointer2[3]])
            center = find_center((pointer1[0], pointer1[1]), (pointer2[0], pointer2[1]))
        elif len(sorted_hand_holds) == 1:
            pointer1 = pointer2 = sorted_hand_holds[0]
            sequence.append(pointer1)
            used_holds.add(pointer1[3])
            center = (pointer1[0], pointer1[1])
        else:
            # No holds to process
            return 0.0, 0.0, 0.0, 0.0, 0.0

    # Step 4: Main loop to build the sequence and calculate weighted distances
    while True:
        # Find the next closest hold
        next_hold = find_next_closest_hold(center, sorted_hand_holds, used_holds, row_weight)
        if not next_hold:
            break

        # Calculate new center point
        new_center = find_center((next_hold[0], next_hold[1]), (pointer2[0], pointer2[1]))

        # Calculate weighted distance for the move
        distance = weighted_distance(center, new_center, row_weight)

        # Retrieve hold magnitudes for current and next center points
        # Current center holds: pointer1 and pointer2
        hold_dir1, hold_mag1 = get_hold_vector(pointer1[1], pointer1[0])  # (row, col)
        hold_dir2, hold_mag2 = get_hold_vector(pointer2[1], pointer2[0])  # (row, col)

        # Next center holds: pointer2 and next_hold
        hold_dir3, hold_mag3 = get_hold_vector(next_hold[1], next_hold[0])  # (row, col)

        # Calculate product of magnitudes
        current_product = hold_mag1 * hold_mag2
        next_product = hold_mag2 * hold_mag3

        # Ensure the product is above the minimum threshold to prevent large scores
        product = current_product * next_product
        if product < min_product:
            product = min_product

        # Calculate difficulty score
        difficulty_score = distance / product

        # Optional: Cap the difficulty_score to a reasonable maximum to prevent outliers
        max_difficulty = 1000.0  # Adjust based on expected difficulty range
        if difficulty_score > max_difficulty:
            difficulty_score = max_difficulty

        # Aggregate weighted distances
        total_weighted_distance += difficulty_score
        weighted_distances.append(difficulty_score)

        # Update pointers
        sequence.append(next_hold)
        used_holds.add(next_hold[3])
        center = new_center
        pointer1 = pointer2
        pointer2 = next_hold

    if not weighted_distances:
        # Avoid division by zero if no moves were made
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Calculate average and maximum weighted distances
    average_weighted_distance = total_weighted_distance / len(weighted_distances)
    highest_weighted_distance = max(weighted_distances)

    # Calculate top N weighted distances
    k = max(1, int(math.ceil(len(weighted_distances) * (percent / 100.0))))
    top_weighted_distances = sorted(weighted_distances, reverse=True)[:k]
    top_n_weighted_distance_sum = sum(top_weighted_distances)
    top_n_average_weighted_distance = top_n_weighted_distance_sum / k if k > 0 else 0.0

    return total_weighted_distance, average_weighted_distance, top_n_weighted_distance_sum, top_n_average_weighted_distance, highest_weighted_distance

# region benchmark 4
def benchmark4(frame, row_weight=0.1, percent=50.0, min_product=0.1):
    def weighted_distance(point1, point2, row_weight=0.1):
        x1, y1 = point1
        x2, y2 = point2
        euclidean = math.hypot(x2 - x1, y2 - y1)
        # Adjust distance based on row_weight
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
            col, row, _ = hold[:3]
            point = (col, row)
            distance = weighted_distance(center, point, row_weight)
            if distance < min_distance:
                min_distance = distance
                next_hold = hold
        return next_hold

    # Step 1: Parse the frame into triplets
    triplet_list = frame_to_triplets(frame)
    frame_words = frame.split('p')[1:]

    # Step 2: Combine triplets with frame words to create (col, row, z, word) tuples
    combined_holds = [(x, y, z, f"p{word}") for triplet, word in zip(triplet_list, frame_words) for x, y, z in [triplet]]

    # Step 3: Categorize holds
    hand_holds = [hold for hold in combined_holds if hold[2] in (0, 1)]
    sorted_hand_holds = sorted(hand_holds, key=lambda x: (x[1], x[0]))
    start_holds = [hold for hold in sorted_hand_holds if hold[2] == 0]

    sequence, used_holds, weighted_distances = [], set(), []
    total_weighted_distance = 0.0

    # Initialize pointers
    if len(start_holds) >= 2:
        pointer1, pointer2 = sorted(start_holds, key=lambda x: x[1])[:2]
        sequence.extend([pointer1, pointer2])
        used_holds.update([pointer1[3], pointer2[3]])
        center = find_center((pointer1[0], pointer1[1]), (pointer2[0], pointer2[1]))
    elif len(start_holds) == 1:
        pointer1 = pointer2 = start_holds[0]
        sequence.append(pointer1)
        used_holds.add(pointer1[3])
        center = (pointer1[0], pointer1[1])
    elif len(sorted_hand_holds) >= 2:
        pointer1, pointer2 = sorted_hand_holds[:2]
        sequence.extend([pointer1, pointer2])
        used_holds.update([pointer1[3], pointer2[3]])
        center = find_center((pointer1[0], pointer1[1]), (pointer2[0], pointer2[1]))
    elif len(sorted_hand_holds) == 1:
        pointer1 = pointer2 = sorted_hand_holds[0]
        sequence.append(pointer1)
        used_holds.add(pointer1[3])
        center = (pointer1[0], pointer1[1])
    else:
        return 0.0, 0.0, 0.0, 0.0, 0.0, []

    # Step 4: Main loop to build the sequence and calculate weighted distances
    while True:
        next_hold = find_next_closest_hold(center, sorted_hand_holds, used_holds, row_weight)
        if not next_hold:
            break

        new_center = find_center((next_hold[0], next_hold[1]), (pointer2[0], pointer2[1]))
        distance = weighted_distance(center, new_center, row_weight)

        # Retrieve hold magnitudes for current and next center points
        hold_mag1 = get_hold_vector(pointer1[1], pointer1[0])[1]
        hold_mag2 = get_hold_vector(pointer2[1], pointer2[0])[1]
        hold_mag3 = get_hold_vector(next_hold[1], next_hold[0])[1]

        # Calculate product of magnitudes
        current_product = hold_mag1 * hold_mag2
        next_product = hold_mag2 * hold_mag3
        product = max(current_product * next_product, min_product)

        # Calculate difficulty score
        difficulty_score = distance / product
        difficulty_score = min(difficulty_score, 1000.0)

        # Aggregate weighted distances
        total_weighted_distance += difficulty_score
        weighted_distances.append(difficulty_score)

        # Update pointers
        sequence.append(next_hold)
        used_holds.add(next_hold[3])
        center = new_center
        pointer1, pointer2 = pointer2, next_hold

    if not weighted_distances:
        return 0.0, 0.0, 0.0, 0.0, 0.0, []

    average_weighted_distance = total_weighted_distance / len(weighted_distances)
    highest_weighted_distance = max(weighted_distances)
    k = max(1, int(math.ceil(len(weighted_distances) * (percent / 100.0))))
    top_weighted_distances = sorted(weighted_distances, reverse=True)[:k]
    top_n_weighted_distance_sum = sum(top_weighted_distances)
    top_n_average_weighted_distance = top_n_weighted_distance_sum / k if k > 0 else 0.0

    return (total_weighted_distance, average_weighted_distance, top_n_weighted_distance_sum,
            top_n_average_weighted_distance, highest_weighted_distance, weighted_distances)
