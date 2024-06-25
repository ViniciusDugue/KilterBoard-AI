# Import necessary libraries
import matplotlib.pyplot as plt
from PIL import Image

# Function to display image and capture click event
def display_image_with_click(image_path):
    # Load the image
    image = Image.open(image_path)
    
    # Adjust figure size before displaying
    plt.figure(figsize=(14, 16))  # Adjust the width and height as needed
    
    # Display the image
    plt.imshow(image)
    plt.title('Click on the image to get pixel coordinates')
    plt.axis('on')  # Ensure the axes are shown for better reference
    
    # Callback function for mouse click event
    def onclick(event):
        if event.button == 1:  # Left mouse button clicked
            # Get the pixel coordinates
            x = int(event.xdata)
            y = int(event.ydata)
            print(f"Clicked on pixel coordinates: ({x}, {y})")
    
    # Connect the callback function to the mouse click event
    plt.connect('button_press_event', onclick)
    
    # Show the plot with interactive features
    plt.show()

# Example usage:
image_path = "KilterAI/KilterBoardSetup.png"  # Replace with your image path
display_image_with_click(image_path)