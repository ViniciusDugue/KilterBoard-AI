import matplotlib.pyplot as plt
from process import map_vgrade, id_to_index, id_to_coordinate, frame_to_ids, frame_to_triplets, triplets_to_matrix, frame_to_sparse_matrix, is_frame_valid, filter_frame, filter_climbs, sort_frame, filtered_df_to_text_file
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import matplotlib.colors as mcolors
import os
from collections import Counter

def plot_vgrade_counts(filtered_df):

    vgrade_counts = filtered_df['vgrade'].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    bars = vgrade_counts.plot(kind='bar', color='skyblue')

    for bar in bars.patches:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f'{int(height)}', ha='center', va='bottom')

    plt.title('Count of Climbs for Each vgrade')
    plt.xlabel('vgrade')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout() 
    plt.show()

def plot_hold_counts(filtered_df):
    filtered_counts = filtered_df['hold_count'].value_counts().sort_index().loc[:35]

    plt.figure(figsize=(14, 6))  # Adjust the width by changing the first value (e.g., 14)
    filtered_counts.plot(kind='bar', color='skyblue')

    plt.title('Count of Climbs with Different Numbers of Holds (Up to 35 Holds)')
    plt.xlabel('Number of Holds')
    plt.ylabel('Count of Climbs')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    total_hold_count = (filtered_counts.index * filtered_counts).sum()
    total_climbs = filtered_counts.sum()
    average_holds = total_hold_count / total_climbs
    
    print("Average Number of Holds:", average_holds)


def analyze_starting_hold_heights(filtered_df, num_climbs=100000):
    starting_hold_height_counts = [0] * 36  # 36 heights from 0 to 35
    climb_counter = 0
    total_starting_hold_heights = 0
    climbs_with_starting_holds_at_35 = 0

    for index, climb in filtered_df.iterrows():
        if climb_counter >= num_climbs:
            break

        try:
            sparse_matrix = frame_to_sparse_matrix(climb['frames'])
            starting_holds_y = sparse_matrix.row[sparse_matrix.data == 4] + 1

            for y in starting_holds_y:
                starting_hold_height_counts[y] += 1
                total_starting_hold_heights += y
                if y == 35:
                    climbs_with_starting_holds_at_35 += 1
            climb_counter += 1
        except (IndexError, ValueError):
            pass 

    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.bar(range(36), starting_hold_height_counts, color='skyblue')
    plt.title('Count of Starting Holds at Each Height')
    plt.xlabel('Height')
    plt.ylabel('Count')
    plt.xticks(range(36))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    total_starting_holds = sum(starting_hold_height_counts)
    average_starting_hold_height = total_starting_hold_heights / total_starting_holds

    percentage_with_starting_holds_at_35 = (climbs_with_starting_holds_at_35 / climb_counter) * 100

    print("Overall Average Starting Hold Height:", average_starting_hold_height)
    print("Percentage of climbs with starting holds at height 35:", percentage_with_starting_holds_at_35)
    return average_starting_hold_height, percentage_with_starting_holds_at_35

def print_unique_climbs_count(filtered_df):
    unique_climbs = filtered_df['name'].nunique()
    print(f"Number of Unique Climbs: {unique_climbs}")

def count_unique_words(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read().strip()  # Read the entire file content and strip any leading/trailing whitespace

        words = content.split()
        word_counts = Counter(words)

        num_unique_words = len(word_counts)
        print("Number of Unique Words:", num_unique_words)

        total_word_count = sum(word_counts.values())
        print("Sum of All Word Counts:", total_word_count)
        
        return word_counts
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except IOError:
        print(f"Error: Unable to read file '{file_path}'.")
        return None

def draw_colored_circles_on_image(image_path, circles, save_as='KilterBoardSetup1.png'):
    hold_colors = {2: '#00FF00', 3: '#00FFFF', 4: '#FF00FF', 5: 'orange'}
 
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for x, y, z in circles:
        color = hold_colors.get(z, 'black')
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline=color)

    save_path = save_as
    image.save(save_path)

    saved_image = Image.open(save_path)
    fig, ax = plt.subplots(figsize=(6, 6)) 
    ax.imshow(saved_image)
    ax.axis('off')
    plt.title('Generated Climb')
    plt.show()

def frame_to_imagecoords(frame):
    ids_list_1, ids_list_2 = frame_to_ids(frame)
    
    triplets = []
    
    for id_1, id_2 in zip(ids_list_1, ids_list_2):
        if 1090 <= id_1 <= 1395: # big holds
            index  = id_1 - 1090
            origin = [45, 910]
            interDistance = 42
            x = origin[0] + ((index % 17) * interDistance) 
            y = origin[1] - ((index // 17) * interDistance)
        
        elif 1073 <= id_1 <= 1089: # bottom large feet
            index  = id_1 - 1073
            origin = [724, 955]
            interDistance = 42
            x = origin[0] - (index * interDistance) 
            y = origin[1] 
        
        elif 1447 <= id_1 <= 1464: # bottom small feet
            index  = id_1 - 1447
            origin = [744, 976]
            interDistance = 42
            x = origin[0] - (index * interDistance) 
            y = origin[1] 

        elif 1465 <= id_1 <= 1599: # small holds
            index  = id_1 - 1465
            if ((index //9) % 2) == 0:# if row is even
                origin = [18, 891]
                interDistance = 86
                x = origin[0] + ((index % 9) * interDistance) 
                y = origin[1] - (((index // 9)/2) * interDistance)
            else: # if row is odd
                origin = [64, 847]
                interDistance = 86  #86
                x = origin[0] + (((index % 9)) * interDistance) 
                y = origin[1] - ((((index // 9)-1) /2) * interDistance)
        
        else:
            # If id_1 does not fall into any of the specified ranges, skip or handle accordingly
            continue
        
        # Create the triplet (x, y, color) and append to the triplets list
        triplet = (x, y, id_2)
        triplets.append(triplet)
    
    return triplets

def frame_to_image(frame):
    image_path = "KilterBoardSetup.png"
    coords_list = frame_to_imagecoords(frame)

    # Add 2 to each id_2 value in the coords_list
    modified_coords_list = [(x, y, id_2 + 2) for (x, y, id_2) in coords_list]

    draw_colored_circles_on_image(image_path, modified_coords_list)
