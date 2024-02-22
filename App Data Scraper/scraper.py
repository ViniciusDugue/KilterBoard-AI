import pyautogui
import h5py
from PIL import Image, ImageGrab
import time
import pygetwindow as gw
import pytesseract
import numpy as np
import re
pytesseract.pytesseract.tesseract_cmd = r'c:\Program Files\Tesseract-OCR\tesseract.exe'
start_time = time.perf_counter()

# Define the bounding box of the region to capture
window = gw.getWindowsWithTitle("BlueStacks App Player")[0]

x_ = window.left
y_ = window.top
width = window.width
height = window.height 
print(x_, y_, width, height)

startingOffset = (30, 357.22)
numclimbs = 25
window.activate()
for climb in range(numclimbs):
    time.sleep(0.1)
    print("---------------------------------------------------")
    # Capture the specified region and load the pixel data
    screenshot = ImageGrab.grab(bbox=(x_+252, y_+130, x_ + 508, y_ + 194))
    # screenshot.show()
    text = pytesseract.image_to_string(screenshot, lang = 'eng')
    print("OCR Result:")
    print(text)
    px = ImageGrab.grab(bbox=(x_, y_, x_ + width, y_ + height)).load()
    matrix = [[0] * 17 for _ in range(19)]
    
    for y in range(19):  # 19
        for x in range(17):
            _x = startingOffset[0] + x * 42.78
            _y = startingOffset[1] + y * 42.78
            colorsample1 = px[_x, _y]
            colorsample2 = px[_x + 28, _y]
            if colorsample1 != (255, 255, 255) and colorsample2 != (255, 255, 255):
                matrix[y][x] = 1
            if x == 16 and y == 1:
                print("Color sample 1:", (_x, _y), colorsample1)
                print("Color sample 2:", (_x + 28, _y), colorsample2)
    
    for row in matrix:
        print(row)
    matrix_flipped = np.flipud(matrix)

    # Create a unique name for the matrix
    lines = text.split('\n')
    filtered_lines = [l for l in text.split('\n') if l]
    # Extract title, parenthesized text, and V0 part
    title = filtered_lines[0]
    setter = f"{filtered_lines[1]}"

    pattern = re.compile(r'(?i)(?<=v)V?(.{2})')
    matches = pattern.findall(filtered_lines[2])
    grade = f'V{"".join([match for match in matches if len(match) == 2])}'
    print()
    if "".join([match for match in matches if len(match) == 2]) == "g ":
        grade = "V8 "
    elif "".join([match for match in matches if len(match) == 2]) == "O " or "".join([match for match in matches if len(match) == 2]) == "o ":
        grade = "V0 "

    # grade = filtered_lines[2].split('/')[1][0:2]  

    matrix_name = f"{grade}-{title}({setter})".replace("\n", "").replace("/","")
    if len(matrix_name) ==0:  # Check if matrix_name is not empty
        matrix_name = f'matrix_{climb}'
    print(matrix_name)
          
    with h5py.File('sparse_matrices.h5', 'a') as hf:
        hf.create_dataset(matrix_name, data=matrix_flipped)
    # Press the "d" key
    print(climb)
    pyautogui.press("q")      

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(elapsed_time, "seconds")
print(1 / elapsed_time, "per second")
print((numclimbs / (1 / elapsed_time)) / 3600, "hours to download")
print((numclimbs / (1 / elapsed_time)) /60 , "minutes to download")
print((numclimbs / (1 / elapsed_time)) , "seconds to download")


