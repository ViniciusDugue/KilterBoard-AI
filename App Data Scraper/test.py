text = """
Destroyer of Worlds
Sealio

4c/VO kw"""

# # Remove all spaces and newlines from the string
# output_string = input_string.replace(" ", "").replace("\n", "")

# print(output_string)
lines = text.split('\n')
filtered_lines = [l for l in text.split('\n') if l]

print(filtered_lines)
line = "4c/VO"

# Split the line based on "/"
parts = line.split('/')[1]
print(parts)