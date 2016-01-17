def getFileList(directory):
    fileList = []
    fileSize = 0
    folderCount = 0
    # For Loop to Cycle Through Directories
    for root, dirs, files in os.walk(directory):
        folderCount += len(dirs)
        for file in files:
            f = os.path.join(root, file)
            fileSize = fileSize + os.path.getsize(f)
            fileList.append(f)
    # Print to Check Data has been located correctly
    print('################################################')
    print('############ Traversing Directories ############\n')
    print('Directory = %s') % (directory)
    print("Total Size is {0} bytes".format(fileSize))
    print('Number of Files = %i') % (len(fileList))
    print('Total Number of Folders = %i') % (folderCount)
    print('################################################\n')
    return fileList
