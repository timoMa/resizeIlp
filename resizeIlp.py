import vigra.impex
import numpy as np
from shutil import copyfile
from h5py import File
import os

from IlastikLabelManager import labelManager

def downsampleProject(ilpPath):
    assert ilpPath.endswith('.ilp'), 'ilp file is required for downsampling. did you mean to upsample?'
    name = ilpPath + 'small.ilp'
    copyfile(ilpPath, name)

    f = File(name)
    for key in f['Input Data/infos'].keys():
        path = f['Input Data/infos/' + key + '/Raw Data/filePath'].value
        rawPath = os.path.split(path)
        rawData = File(rawPath[0], 'r')
        rawShape = rawData['im'].shape
        targetShape = tuple([int(ordinate / 2) for ordinate in rawShape])

        labels = labelManager(name)
        labeledBlocks = labels.getSubBlocks()

        # find out the dimension of the block, there should be a better way of doing that.
        offsets = np.array([labeledBlock[0] for labeledBlock in labeledBlocks])
        shapes = np.array([labeledBlock[1].shape[:3] for labeledBlock in labeledBlocks])
        data = [labelsBlock[1][:,:,:,0] for labelsBlock in labeledBlocks]

        # write all labeles into one big array
        labelBlockTotal = np.zeros(rawShape, dtype=np.uint8)
        for offset, shape, dataBlock in zip(offsets, shapes, data):
            index = [slice(offset[0], offset[0] + shape[0]),
                    slice(offset[1], offset[1] + shape[1]),
                    slice(offset[2], offset[2] + shape[2])]
            labelBlockTotal[index] += dataBlock

        resized = np.zeros(targetShape)


        resized = np.zeros(targetShape, dtype=np.uint8)
        vigra.graphs.downsampleLabels(labelBlockTotal, int(labelBlockTotal.max()), 0.1, resized)

        labels.clear()
        labels.flush()

        step = 90
        exportBlocks = []
        offsets = []

        resizedShape = resized.shape
        for x in np.arange(0, resizedShape[0], step):
            for y in np.arange(0, resizedShape[1], step):
                for z in np.arange(0, resizedShape[2], step):
                    exportBlock = resized[x:x+step, y:y+step, z:z+step]
                    exportBlock = exportBlock[:,:,:,None]
                    # only write labels, if there are some
                    # if exportBlock.max() != 0
                    offset = [x,y,z]
                    labels.addBlockLabel(exportBlock, offset)

        print 'resize raw data'
        rawData = vigra.readHDF5(*rawPath)

        exportRawPath = [rawPath[0].split('.')[0] + '_resized.h5', rawPath[1]]

        rawSmall = vigra.sampling.resize(rawData.astype(np.float32), resizedShape, 0)
        vigra.writeHDF5(rawSmall, *exportRawPath)


        try:
            del f['Input Data/infos/' + key + '/Raw Data/filePath']
        except KeyError:
            pass

        f['Input Data/infos/' + key + '/Raw Data/filePath'] = exportRawPath[0] + '/' + exportRawPath[1]

def upsample(path):
    """ upsampling of multichannel data (ignoring the last dimension which should be used for channels)"""
    splitPath = path.split('.h5')
    assert len(splitPath) != 1, "file seems to not have the .h5 extension"

    data = vigra.readHDF5(splitPath[0] + '.h5', splitPath[1]).squeeze()
    assert len(data.shape) == 3, "only 3d data supported"

    if isinstance(data,vigra.VigraArray):
        data = data.view(np.ndarray)

    data = np.require(data, dtype=np.float32)
    data = vigra.sampling.resize(data, shape=[size*2 for size in data.shape[:3]], order=2)
    vigra.writeHDF5(data, splitPath[0] + '_upsampled.h5', splitPath[1])


if __name__ == '__main__':
    import argparse
    paraser = argparse.ArgumentParser()
    paraser.add_argument("input_path")
    paraser.add_argument("--upsample", action="store_true")

    args = paraser.parse_args()
    path = args.input_path
    if args.upsample:
        upsample(path)
    else:
        downsampleProject(path)
    # upsampleMultiChannelData('/home/timo/multiscaletest/results/actualHoles_resized_probs.h5', 'exported_data')
