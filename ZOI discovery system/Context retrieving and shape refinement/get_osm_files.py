import os

# Function to find all files with a specific extension in a folder and its subfolders
def find_files_with_extension(base_folder, extension):
    # List to store the paths of files with the specified extension
    osm_files = []

    # Traverse the folder and its subfolders
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            # Check if the file has the specified extension
            if file.endswith(extension):
                # Get the full path of the file and add it to the list
                osm_files.append(os.path.join(root, file))
    
    return osm_files

