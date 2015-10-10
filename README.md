## Usage
downsampling:

python resizeIlp.py <pathToILP>.ilp

upsampling:

    python resizeILP.py <pathToProbaabilityHdf5>.h5/intermalPath --upsample

upsampling and stacking with raw data:

    python resizeIlp.py <pathToProbabilityHdf5>.h5/<internalPath> --upsample --stackWithRaw <pathToRawHdf5>.h5/<internalPath>
