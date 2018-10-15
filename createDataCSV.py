import csv
import os
import absl.app as app

def main(argv):
    with open('tifData.csv', mode='w') as f:
        writer = csv.writer(f, delimiter=',')
        # separate columns for signal and target
        writer.writerow(['path_tif_signal', 'path_tif_target', 'num_z_slices'])
        # sort the list of files
        allFiles = sorted(os.listdir('data/tiff/'))
        # filename prefix 
        currPrefix = ''
        # count the number of Z slices
        zCount = 0
        # list of rows to write to file
        sliceMap = dict()
        # loop over files to create a dictionary of the slices per file
        for filename in allFiles:
            start = filename.find('z')
            newPrefix = filename[:start]
            if newPrefix != currPrefix:
                currPrefix = newPrefix
                zCount = 1
            else:
                zCount += 1
            sliceMap[newPrefix] = zCount
        
        for filename, zCount in sliceMap.items():
            if 'c2' in filename and zCount == sliceMap[filename.replace('c2', 'c1')]:
                writer.writerow(['data/tiff/' + filename, 'data/tiff/' + filename.replace('c2', 'c1'), zCount])

if __name__ == "__main__":
    app.run(main)
