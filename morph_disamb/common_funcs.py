import os


def get_file_paths(directory, file=None):
    """
    Collects the file paths from the given directory if the file is not given, otherwise creates a path
    joining the given directory and file.
    :param directory: The directory where the file(s) can be found
    :param file: A file in the directory
    :return: The sorted list of collected file paths
    """
    file_paths = []

    # get the absolute path of the one given file
    if file is not None:
        source_file_path = os.path.join(directory, file)
        if os.path.isfile(source_file_path):
            file_paths.append(source_file_path)

    # if the given doesn't exist or it wasn't given, all files from the directory will be loaded
    # except desktop.ini
    if len(file_paths) == 0:
        for child in os.listdir(directory):
            if child != 'desktop.ini':
                child_path = os.path.join(directory, child)
                if os.path.isfile(child_path):
                    file_paths.append(child_path)

    return sorted(file_paths)


if __name__ == '__main__':
    pass
