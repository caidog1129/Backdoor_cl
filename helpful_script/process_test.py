import os
import shutil
import argparse

# Function to create a folder for each file and move the file inside the folder
def organize_files(directory):
    if not os.path.isdir(directory):
        print(f"The directory '{directory}' does not exist.")
        return

    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            folder_name, _ = os.path.splitext(filename)
            folder_path = os.path.join(directory, folder_name)

            # Create the folder if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)

            # Move the file into the folder
            shutil.move(os.path.join(directory, filename), os.path.join(folder_path, filename))

# Function to evenly distribute folder names into instance.txt files with absolute paths
def evenly_distribute_folders(directory):
    if not os.path.isdir(directory):
        print(f"The directory '{directory}' does not exist.")
        return

    folders = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]
    abs_directory = os.path.abspath(directory)  # Get the absolute path of the directory

    # Calculate how many folders should be placed in each instance.txt file
    folders_per_instance = len(folders) // 2

    for i in range(2):
        start_idx = i * folders_per_instance
        end_idx = (i + 1) * folders_per_instance if i < 1 else None
        instance_folders = folders[start_idx:end_idx]

        instance_file_path = f"ins_{i + 1}.txt"  # Define the path for the instance.txt file in the current directory

        with open(instance_file_path, "w") as instance_file:
            # Write absolute paths to instance.txt
            for folder in instance_folders:
                abs_folder_path = os.path.join(abs_directory, folder)
                abs_folder_path = abs_folder_path.replace("home", "nethome")
                instance_file.write(abs_folder_path + "\n")

def main():
    parser = argparse.ArgumentParser(description="Organize files in a directory and distribute folder names into instance.txt files with absolute paths.")
    parser.add_argument("directory", help="The directory to process")

    args = parser.parse_args()

    organize_files(args.directory)
    evenly_distribute_folders(args.directory)

    print("Folders organized, and instance.txt files with absolute paths created in the current directory.")

if __name__ == "__main__":
    main()
