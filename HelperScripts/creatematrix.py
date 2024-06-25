# import numpy as np

# # Create a 3D matrix filled with zeros
# matrix = np.zeros((35, 35, 2))

# # Save the matrix to a .npz file
# np.savez('hold_embeddings.npz', matrix=matrix)

# loaded_data = np.load('hold_embeddings.npz')
# loaded_matrix = loaded_data['matrix']

# # Iterate over each element and print it
# for row in loaded_matrix:
#     print(','.join(map(str, row)))

# Example of how to write a 35x35x2 matrix to a text file with lists
# with open('holdembeddings.txt', 'w') as f:
#     for i in range(35):
#         # Write each vector in the row separated by commas
#         for j in range(35):
#             # Write the 2x1 vector [0.0, 0.0]
#             f.write("[0.0, 0.0]")
#             if j < 34:
#                 f.write(", ")  # Add a comma between vectors
#         # Add a newline after each row
#         f.write("\n")

import numpy as np

# Create an empty .npz file
np.savez('holdembeddings.npz')

# Read the values from the text file
matrix = []

# Convert the matrix to a NumPy array
matrix_array = np.array(matrix)
print(matrix_array)

# Save the matrix to an npz file
np.savez('holdembeddings.npz', matrix=matrix_array)

