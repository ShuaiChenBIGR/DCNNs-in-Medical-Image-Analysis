from medpy import metric
import pandas as pd
import SimpleITK as sitk
import numpy as np
import Modules.Common_modules as cm
import subprocess, os, shutil

ITKToolsBinDir = "/home/schen/ItkTools/bin/bin"


def SegmentDist(File1, File2, OutDr, name):

  cm.mkdir(OutDr + '/Temp')
  cmd1 = subprocess.getoutput(ITKToolsBinDir + '/pxsegmentationdistance -in ' + File1 + ' ' + File2 + ' -car true -out ' + OutDr + '/Temp' + '/out.mhd')
  cmd2 = subprocess.getoutput(
    ITKToolsBinDir + '/pxunaryimageoperator -in ' + OutDr + '/Temp' + '/outDIST.mhd -ops ABS -out ' + OutDr + '/Temp' + '/outDISTabs.mhd')
  cmd3 = subprocess.getoutput(
    ITKToolsBinDir + '/pxstatisticsonimage -in ' + OutDr + '/Temp' + '/outDISTabs.mhd -s arithmetic -mask ' + OutDr + '/Temp' + '/outEDGE.mhd')
  shutil.rmtree(OutDr + '/Temp')

  # text_file = open(OutDr + '/surface_distance_' + name + '.txt', 'w')
  # text_file.write(str(cmd1))
  # text_file.close()
  #
  # text_file = open(OutDr + '/surface_distance_' + name + '.txt', 'a')
  # text_file.write(str(cmd2))
  # text_file.close()

  text_file = open(OutDr + '/surface_distance_' + name + '.txt', 'w')
  text_file.write(str(cmd3))
  text_file.close()

  return [cmd1, cmd2, cmd3]

# reference_segmentation = sitk.ReadImage('tumorSegm', sitk.sitkUInt8)
# segmentation = sitk.ReadImage('tumorSegm2',sitk.sitkUInt8)
#
#
# class SurfaceDistanceMeasuresITK(Enum):
#     hausdorff_distance, max_surface_distance, avg_surface_distance, median_surface_distance, std_surface_distance = range(5)
#
# class MedpyMetricDists(Enum):
#     hausdorff_distance, avg_surface_distance, avg_symmetric_surface_distance = range(3)
#
#
# surface_distance_results = np.zeros((1,len(SurfaceDistanceMeasuresITK.__members__.items())))
# surface_dists_Medpy = np.zeros((1,len(MedpyMetricDists.__members__.items())))
# segmented_surface = sitk.LabelContour(segmentation)
#
# # init signed mauerer distance as reference metrics
# reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(reference_segmentation, squaredDistance=False, useImageSpacing=True))
#
# label_intensity_statistics_filter = sitk.LabelIntensityStatisticsImageFilter()
# label_intensity_statistics_filter.Execute(segmented_surface, reference_distance_map)
#
# hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
# hausdorff_distance_filter.Execute(reference_segmentation, segmentation)
#
# surface_distance_results[0,SurfaceDistanceMeasuresITK.hausdorff_distance.value] = hausdorff_distance_filter.GetHausdorffDistance()
# surface_distance_results[0,SurfaceDistanceMeasuresITK.max_surface_distance.value] = label_intensity_statistics_filter.GetMaximum(label)
# surface_distance_results[0,SurfaceDistanceMeasuresITK.avg_surface_distance.value] = label_intensity_statistics_filter.GetMean(label)
# surface_distance_results[0,SurfaceDistanceMeasuresITK.median_surface_distance.value] = label_intensity_statistics_filter.GetMedian(label)
# surface_distance_results[0,SurfaceDistanceMeasuresITK.std_surface_distance.value] = label_intensity_statistics_filter.GetStandardDeviation(label)
#
# surface_distance_results_df = pd.DataFrame(data=surface_distance_results, index = list(range(1)),
#                               columns=[name for name, _ in SurfaceDistanceMeasuresITK.__members__.items()])
#
# img_array = sitk.GetArrayFromImage(reference_segmentation)
# seg_array = sitk.GetArrayFromImage(segmentation)
# # reverse array in the order x, y, z
# img_array_rev = np.flip(img_array,2)
# seg_array_rev = np.flip(seg_array,2)
# vxlspacing = segmentation.GetSpacing()
#
# surface_dists_Medpy[0,MedpyMetricDists.hausdorff_distance.value] = metric.binary.hd(seg_array_rev,img_array_rev, voxelspacing=vxlspacing)
# surface_dists_Medpy[0,MedpyMetricDists.avg_surface_distance.value] = metric.binary.asd(seg_array_rev,img_array_rev, voxelspacing=vxlspacing)
# surface_dists_Medpy[0,MedpyMetricDists.avg_symmetric_surface_distance.value] = metric.binary.assd(seg_array_rev,img_array_rev, voxelspacing=vxlspacing)
#
# surface_dists_Medpy_df = pd.DataFrame(data=surface_dists_Medpy, index = list(range(1)),
#                               columns=[name for name, _ in MedpyMetricDists.__members__.items()])
