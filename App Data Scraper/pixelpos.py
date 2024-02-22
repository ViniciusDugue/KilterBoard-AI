import pyautogui
import keyboard
from PIL import Image, ImageGrab
import pygetwindow as gw

window = gw.getWindowsWithTitle("BlueStacks App Player")[0]
x = window.left
y = window.top
width = window.width
height = window.height

px = ImageGrab.grab(bbox=(x, y, x + width, y + height)).load()

x_coordinates = []
y_coordinates = []

def measure_position(event):
    if event.event_type == keyboard.KEY_DOWN and event.name == 'w':
        cursor_x, cursor_y = pyautogui.position()
 
        x_coordinates.append(cursor_x)
        y_coordinates.append(cursor_y)
        
        print(f"Cursor coordinates: ({cursor_x}, {cursor_y}) Color: {px[cursor_x, cursor_y]}")
    elif event.event_type == keyboard.KEY_DOWN and event.name == 'space':

        input_x = float(input("Enter x coordinate: "))
        input_y = float(input("Enter y coordinate: "))

        print(f"Color at ({input_x}, {input_y}): {px[input_x, input_y]}")

keyboard.on_press(measure_position)

keyboard.wait('esc')  

x_differences = [x_coordinates[i + 1] - x_coordinates[i] for i in range(len(x_coordinates) - 1)]
y_differences = [y_coordinates[i + 1] - y_coordinates[i] for i in range(len(y_coordinates) - 1)]

average_x_difference = sum(x_differences) / len(x_differences)
average_y_difference = sum(y_differences) / len(y_differences)

print(f"Average distance between cursor clicks in the X direction: {average_x_difference}")
print(f"Average distance between cursor clicks in the Y direction: {average_y_difference}")
