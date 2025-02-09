from .sort_frame import *
from .process import *
#this file is necessary due to circular import issues between process.py and sort_frame.py

# region dataframe
def create_filtered_climbs_df(climbs, climb_stats, vgrade=-1, angle=-1):
    # Merge climbs and climb stats dataframes
    merged_df = pd.merge(climbs, climb_stats, left_on='uuid', right_on='climb_uuid', how='inner')

    # Apply base filters
    merged_df = merged_df[(merged_df['layout_id'] == 1) & (merged_df['ascensionist_count'] >=1)]

    # Map vgrade and select relevant columns
    merged_df['vgrade'] = merged_df['display_difficulty'].apply(map_vgrade)
    filtered_columns = ['name', 'vgrade', 'angle_y', 'display_difficulty', 'ascensionist_count', 'frames', 'is_draft', 'climb_uuid']
    filtered_df = merged_df.loc[:, filtered_columns].drop_duplicates(subset=['name'])

    # Count holds and filter climbs by hold count
    filtered_df['hold_count'] = filtered_df['frames'].str.count('p')
    filtered_df = filtered_df[filtered_df['hold_count'] <= 21]

    # Apply vgrade filter
    if vgrade != -1:
        if vgrade == 0:
            filtered_df = filtered_df[filtered_df['vgrade'].isin([0, 1])]
        else:
            vgrade_range = [max(vgrade - 1, 0), min(vgrade + 1, 15)]
            filtered_df = filtered_df[filtered_df['vgrade'].between(vgrade_range[0], vgrade_range[1])]

    # Apply angle filter
    if angle != -1:
        filtered_df = filtered_df[filtered_df['angle_y'] == angle]

    # Apply frame validation filter
    filtered_df = filtered_df[filtered_df['frames'].apply(is_frame_valid)]

    # Sort frames and save to text file
    
    filtered_df['frames'] = filtered_df['frames'].apply(sort_frame_4)
    return filtered_df

def filtered_df_to_text_file(filtered_df, file_path='climbs.txt'):
    climb_frames = filtered_df['frames'].apply(lambda x: ' '.join(x.split('p')))

    with open(file_path, 'w') as file:
        for climb_frame in climb_frames:
            file.write(climb_frame + '\n')