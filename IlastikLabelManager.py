from h5py import File
import numpy as np

def pointsToPosition(start, stop, invert=True):
    pos = '['
    if invert:
        for startCoord, stopCoord in zip(start[::-1], stop[::-1]):
            pos += str(startCoord) + ':' + str(stopCoord) + ','
    else:
        for startCoord, stopCoord in zip(start, stop):
            pos += str(startCoord) + ':' + str(stopCoord) + ','

    # delete the last space and comma and make a closing bracket sign instead.
    pos += '0:1]'
    return pos


def strToPos(posString):
    posString = posString.split(':')
    x = int(posString[0][1:])
    y = int(posString[1].split(',')[1])
    z = int(posString[2].split(',')[1])
    return [x,y,z]

class labelManager(object):

    def __init__(self, fileName, startBlockNum = 0):
        self._f = File(fileName,'r+')
        self._blockNumber = startBlockNum
        self._maxLabelNum = 9999

    def addBlockLabel(self, data, start, stop=None, invert = False):
        if not stop:
            stop = [length + offset for length, offset in zip(data.shape, start)]

        if self._blockNumber <= self._maxLabelNum:
            dataset = self._f['PixelClassification/LabelSets/labels000'].create_dataset('block%04d' % self._blockNumber, data=(data.astype(np.uint8)))
            dataset.attrs.create('blockSlice',pointsToPosition(start, stop, invert))
            self._blockNumber += 1
        else:
            print 'Warning: maximum label block number exceeded. Unable to add further labels.'


    def addMultipleSingleLabels(self, positions, labelValue):
        for point in positions.T:
            self.addLabels(labelValue, pointsToPosition(point, point+1))

    def addSingleLabel(self, labelValue, position):
        dataset = self._f['PixelClassification/LabelSets/labels000'].create_dataset('block%04d' % self._blockNumber, data=[[[[np.uint8(labelValue)]]]])
        dataset.attrs.create('blockSlice',position)
        self._blockNumber += 1

    def clear(self):
        dataset = self._f['PixelClassification/LabelSets/labels000']
        for key in dataset.keys():
            del dataset[key]
        self._blockNumber = 0

    def getSubBlocks(self):
        """ returns subblocks containing the labels together with their corresponding offsets"""

        dataset = self._f['PixelClassification/LabelSets/labels000']
        labelBlocks = []
        for key in dataset:
            offset = strToPos(dataset[key].attrs.get('blockSlice'))
            values = dataset[key].value
            labelBlocks.append([offset, values])
            print key
        return labelBlocks

    def getInSingleBlock(self, shape=None):
        """ returns a block containing all the labels. The return is guaranteed to start at (0,0,0) global coordinates,
        it may however not cover the whole block (max(shape[0]), max(shape[1]), max(shape[2])), since there is no good way
        of determining the shape of the raw data from ilasti"""


        # get the labels as they are saved in the projecct
        labeledBlocks = self.getSubBlocks()

        offsets = np.array([labeledBlock[0] for labeledBlock in labeledBlocks])
        shapes = np.array([labeledBlock[1].shape[:3] for labeledBlock in labeledBlocks])
        data = [labelsBlock[1][:,:,:,0] for labelsBlock in labeledBlocks]

        if shape is None:
            # find out the dimension of the block, there should be a better way of doing that.
            shape = np.max(offsets + shapes[:,:3], axis=0)

        # write all labeles into one big array
        labelBlockTotal = np.zeros(shape, dtype=np.uint8)
        for offset, shape, dataBlock in zip(offsets, shapes, data):
            index = [slice(offset[0], offset[0] + shape[0]),
                    slice(offset[1], offset[1] + shape[1]),
                    slice(offset[2], offset[2] + shape[2])]
            labelBlockTotal[index] += dataBlock

        return labelBlockTotal


    def flush(self):
        self._f.flush()

    def changeRawDataPath(self, newPath):
        """ deletes all saved paths and replaces it with the path 'newPath' """
        dataset = self._f['Input Data/infos/lane0000/Raw Data/']
        dataset.pop('filePath')
        dataset.create_dataset('filePath', data=newPath)

