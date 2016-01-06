import vigra.impex
import numpy as np
from shutil import copyfile
from h5py import File
import os

from libs.essentials import splitPath

from IlastikLabelManager import labelManager

def downsampleProject(ilpPath, adjustScales=False, resizeRaw=True):
    assert ilpPath.endswith('.ilp'), 'ilp file is required for downsampling. did you mean to upsample?'
    name = ilpPath + 'small.ilp'
    copyfile(ilpPath, name)

    f = File(name)
    # move feature matrix to smaller sigmas (first row should stay the same, second should not be overwritten, but 'or'ed, the rest is just moved to the next smaller sigma.
    if adjustScales:
        matrix = f['FeatureSelections/SelectionMatrix/'].value
        matrix[:,1] = np.logical_or(matrix[:,2], matrix[:,1])
        matrix[:,2:-1] = matrix[:,3:]
        matrix[:,-1] = False
        del f['FeatureSelections/SelectionMatrix/']
        f['FeatureSelections/SelectionMatrix/'] = matrix


    for key in f['Input Data/infos'].keys():
        path = f['Input Data/infos/' + key + '/Raw Data/filePath'].value

        # resize the labels
        labels = labelManager(name)
        labeledBlocks = labels.getSubBlocks()
        labeledBlocksResized = list()
        for labeledBlock in labeledBlocks:
            targetShape = tuple([int(ordinate / 2) for ordinate in labeledBlock[1].shape[:3]])
            resized = np.zeros(targetShape, dtype=np.uint8)
            vigra.graphs.downsampleLabels(labeledBlock[1], int(labelBlockTotal.max()), 0.1, resized)
            offsetResized = tuple([int(ordinate/2) for ordinate in labeledBlock[0])
            labeledBlocksResized.append((offsetResized, resized)

        # clear the old labelds
        labels.clear()
        labels.flush()

        for labeledBlock in labeledBlocksResized:
            labels.addBlockLabel(labeledBlock[1][:,:,:,None], labeledBlock[0])

        labels.flush()

        if resizeRaw:
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

def upsample(probPath, rawPath):
    """ upsampling of multichannel data of an ilp (ignoring the last dimension which should be used for channels) and concatanating int with the raw image """

    splitProbPath = probPath.split('.h5')
    assert len(splitProbPath) !=1, "file " + splitProbPath[0] + "seems to not have the .h5 extension"

    data = vigra.readHDF5(splitProbPath[0] + '.h5', splitProbPath[1]).squeeze()

    if len(data.shape) != 4:
        print "WARNING: untested for data other than 3d + channel."

    if isinstance(data,vigra.VigraArray):
        data = data.view(np.ndarray)

    # normalize probabilities and save them as hdf5
    data = np.require(data, dtype=np.float32)
    data = vigra.sampling.resize(data, shape=[size*2 for size in data.shape[:-1]], order=2)
    data -= data.min()
    data *= 255 / data.max()
    data = data.astype(np.uint8)

    if rawPath != None:
        splitRawPath = rawPath.split('.h5')
        raw = vigra.readHDF5(splitRawPath[0] + '.h5', splitRawPath[1]).squeeze()
        assert len(splitRawPath) !=1, "file " + splitRawPath[0] + "seems to not have the .h5 extension"
        data = np.concatenate((raw[:,:,:,None], data), axis=3)

    vigra.writeHDF5(data.astype(np.uint8), splitProbPath[0] + '_upsampled.h5', splitProbPath[1])


def pureConcaternation(probPath, rawPath):
    """concatanating int with the raw image """

    assert len(splitPath(probPath)) == 2, "file " + probPath + "seems to not have the .h5 extension"

    data = vigra.readHDF5(*splitPath(probPath)).squeeze()

    if len(data.shape) != 4:
        print "WARNING: untested for data other than 3d + channel."

    if isinstance(data,vigra.VigraArray):
        data = data.view(np.ndarray)

    # normalize probabilities and save them as hdf5
    data = np.require(data, dtype=np.float32)
    data = vigra.sampling.resize(data, shape=[size*2 for size in data.shape[:-1]], order=2)
    data -= data.min()
    data *= 255 / data.max()
    data = data.astype(np.uint8)

    if rawPath != None:
        assert len(splitPath(rawPath)) == 2, "file " + rawPath + "seems to not have the .h5 extension"
        raw = vigra.readHDF5(*splitPath(rawPath)).squeeze()
        data = np.concatenate((raw[:,:,:,None], data), axis=3)

    vigra.writeHDF5(data.astype(np.uint8), splitProbPath[0] + '_upsampled.h5', splitProbPath[1])



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("--stackWithRaw", default=None)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--adjustScales", action="store_true")

    args = parser.parse_args()
    path = args.input_path
    if args.upsample:
        upsample(path, args.stackWithRaw)
    elif args.stackWithRaw is not None:
        pureConcaternation(probPath, rawPath)
    else:
        downsampleProject(path, args.adjustScales)
    # upsampleMultiChannelData('/home/timo/multiscaletest/results/actualHoles_resized_probs.h5', 'exported_data')
