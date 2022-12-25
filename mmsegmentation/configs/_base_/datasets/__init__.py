from .potsdam import PotsdamDataset
from .stare import STAREDataset
#from .voc import PascalVOCDataset
from .suadd23 import SUADDDataset
 
__all__ = [
     'CustomDataset', 'build_dataloader', 'ConcatDataset', 'RepeatDataset',
     'PascalContextDataset59', 'ChaseDB1Dataset', 'DRIVEDataset', 'HRFDataset',
     'STAREDataset', 'DarkZurichDataset', 'NightDrivingDataset',
     'COCOStuffDataset', 'LoveDADataset', 'MultiImageMixDataset',
#    'iSAIDDataset', 'ISPRSDataset', 'PotsdamDataset', 'FaceOccludedDataset'
   'iSAIDDataset', 'ISPRSDataset', 'PotsdamDataset', 'FaceOccludedDataset', 'SUADDDataset'
 ]