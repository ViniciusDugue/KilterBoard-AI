# from PIL import ImageGrab

# # Capture the screen and load the pixel data
# px = ImageGrab.grab().load()

# # Iterate over pixels in a grid pattern
# for y in range(0, 100, 10):
#     for x in range(0, 100, 10):
#         # Get the color value of the pixel at (x, y)
#         color = px[x, y]
#         print(f"Color value at ({x}, {y}): {color}")

from PIL import Image, ImageGrab
import time
import pygetwindow as gw
start_time = time.perf_counter()


# Define the bounding box of the region to capture
window = gw.getWindowsWithTitle("BlueStacks App Player")[0]
x = window.left
y = window.top #+ 127
width = window.width #-30
height = window.height #-127 -200

# Capture the specified region and load the pixel data
px = ImageGrab.grab(bbox=(x, y, x + width, y + height)).load()

# Create a new image with the same size as the captured region
image = Image.new("RGB", (width, height))

# Iterate over pixels in the captured region and set the color of each pixel in the new image
for y in range(height):
    for x in range(width):
        color = px[x, y]
        # print(f"Color value at ({x}, {y}): {color}")
        image.putpixel((x, y), color)

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(elapsed_time, "seconds")
print(1/elapsed_time, "per second")
print((100000/(1/elapsed_time))/3600, "hours to download")
# Save the image to a file (you can specify the de sired file format)
image.save("captured_image.png")
