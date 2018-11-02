import glob
from os.path import split
import csv


def get_files_directory(path,full_filenames=True):
    '''
    Inputs:
    Given a path it will return a list of all files as a list
    path can include wild cards see example
    full_filenames =True joins filenames to path
    
    Returns: 
    List of filenames with or without paths
    
    Example:
    file_list = get_files_directory('~/test*.png')
    
    
    '''
    filename_list = glob.glob(path)
    if full_filenames==True:
        return filename_list
    else:
        f = [split(f)[1] for f in filename_list]
        return f
    
def write_row_to_csv(filename,list_for_row,append=True):
    '''
    Inputs:
    filename - includes full path
    list_for_row - is a single 1d list
    append - False creates new file.
    
    Returns:
    True if successfully written
    '''
    
    if append == True:
        mode = 'a'
    else:
        mode = 'w'
    with open(filename,mode)as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|')
        writer.writerow(list_for_row)
    return True