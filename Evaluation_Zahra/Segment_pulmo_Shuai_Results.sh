#!/bin/bash
#this script does the following steps:
# 1. cropes the Manual and Automatic Segmentation (initialization by Shuai with deep learning) accoding to the manual Centerline and saves them both in mhd and dcm 
# 2. Applies opfront
# 3. computesdice jaccard MSD
# 4. Diameter per slice based on bifurcation point from Zahra results
# 18-12-2017
Start_Total=$(date +%s)

# Defin Paths and Directories:
Path_Opfront=/home/zsedghi/Medical_Phisics_AortaPulmoSegmentation/Scripts/Registeration/Opfront_OnlySegmentation.sh
Matlab_path=/home/zsedghi/Medical_Phisics_AortaPulmoSegmentation/Matlab
Path_Cost=/home/zsedghi/Medical_Phisics_AortaPulmoSegmentation/Scripts/Cost_local_NOManualCompare.sh
Path_MeanDIST=/home/zsedghi/Scripts/Original/MeanSurfaceDist.sh

Vol_dcm=$1				#e.g. Vol_dcm=/scratch/zsedghi/Data/DLCST/vol4.dcm  
Auto_Initialization=$2	#e.g. Auto_Initialization=/scratch/zsedghi/Shuai_Results/DeepLearning_Output/vol4/masksTestPredicted.dcm
Manual_Segmentation=$3  #e.g. Pulmonary artery  manual segmentation (dcm)   /vol4/vol4_Manual_3DMask.dcm
Manual_LeftCenterline=$4  	#e.g. LEft Pulmonary artery  manual resampled centerline (txt) /vol4/Pulmonary artery /vol4_leftPulmonary_Centerline_resampled.txt
Manual_RightCenterline=$5 	#e.g. Right Pulmonary artery  manual resampled centerline (txt) /vol4/Pulmonary artery /vol4_RightPulmonary_Centerline_resampled.txt
Manual_Bif=$6			#e.g. Manual_Bif=/scratch/zsedghi/Data/DLCST_BifPoint/vol4_BifurcationPoint.txt
Output_folder=$7		#e.g. Output_folder=/scratch/zsedghi/Shuai_Results/Segmentation/vol4
      
if [ $# != "7" ]	 # countes the number of inputs and if less than 9 it gives error
then
	 echo "  "
	 echo " Not enough inputs."
	 echo "  "
	 echo "6 Inputs are required which should be in this order: 
	 1. Full volume (dcm) 
	 2.initialization for opfront (output of deep learning)
	 3.Manual Pulmonary artery  Segmentation( dcm format)
	 4.Manual left Pulmonary artery  Centerline (resampled txt)
	 5.Manual Right Pulmonary artery  Centerline (resampled txt)
	 6.Manual pulmonary artery bifurcation point(txt file) 
	 7.Output folder  (a directory not a file name)
	 " 
	 exit
fi

## Creat the outputs and log file
File_Base=$(basename "${Vol_dcm}" .dcm) 
File=${File_Base%_*} 
Log_Folder=$Output_folder/Logs
NOW=$(date +%d-%m-%Y)

 /bin/echo "  " 
 /bin/echo " Segmentation with graph cut , initialized with deeplearning output from Shuai" 
 /bin/echo "Script name: /home/zsedghi/Medical_Phisics_AortaPulmoSegmentation/Scripts/Deeplearning/Segment_Shuai_Results.sh" 
 /bin/echo "Input Volume: $Vol_dcm" 
 /bin/echo "Input Volume name: $File" 
 /bin/echo "Output Folder: $Output_folder"
 /bin/echo "$(date +%d-%m-%Y----%T)" 

###########################################################################
############ Create the output folders ####################################
 Seg_Pulmo_Org=$Output_folder/Pulmo_Segmentation_OriginalVolume
 Seg_Pulmo_Contrast=$Output_folder/Pulmo_Segmentation_ContrastReduced
 Seg_Net=$Output_folder/Deeplearning_Accuracy

 mkdir -p $Seg_Pulmo_Org
 mkdir -p $Seg_Pulmo_Contrast
 mkdir -p $Seg_Net
 
#################################################################################################################################################
### 1. Cut manual and initial segmentation and get the initialization accuracy(DSC, MSD) All based on manual centerline #########################
 /bin/echo "********************************************************************************************************************************************" 
 /bin/echo "**** 1. Cut manual and initial segmentation and get the initialization accuracy(DSC, MSD) All based on manual centerline***" 
 /bin/echo "********************************************************************************************************************************************"  
 /bin/echo " 1.a) Cut both segmentations and calculate dice and jaccard coefficient: "
 module load mcr;
 $Matlab_path/Dice_Correct_Pulmo_ShuaiResults $Manual_LeftCenterline $Manual_RightCenterline $Manual_Segmentation $Auto_Initialization $Seg_Net/Manual $Seg_Net/Auto

 /bin/echo " 1.b) calculate the mean surface distance (mean, max, min, std of the surface distance): "
 $Path_MeanDIST $Seg_Net/Auto.mhd $Seg_Net/Manual.mhd $Seg_Net/MSD_CompleteVolume.txt
 rm $Seg_Net/*.mhd 
 rm $Seg_Net/*.raw
 
 /bin/echo Dice, Mean surface distance for the initialaization Ends on: `date`

 ###################################################################################################################
 ########### 2. Pulmonary artery  segmentation initialized with the deep learning results on original volume ###################
 /bin/echo  "********************************************************************************************************" 
 /bin/echo  "***** 2.  Pulmonary artery  segmentation initialized with the deep learning results on original volume *************"  
 #Parameters_Aorta:
 inner_smooth_P="50"  
 grid_spacing_P="0.8"  	
 kernel_size_P="31" 
 sample_interval_P="0.5" 
 edge_length_P="0.8"  
 regularization_P="1.6"  
 Max_inner_flowLine_P="14"  
 Max_Outer_flowLine_P="20"
 
 /bin/echo "Pulmonary artery  parameters are:
 inner_smooth_P: $inner_smooth_P
 grid_spacing_P: $grid_spacing_P    
 kernel_size_P: $kernel_size_P    
 sample_interval_P: $sample_interval_P 
 edge_length_P: $edge_length_P     
 regularization_P: $regularization_P   
 Max_inner_flowLine_P: $Max_inner_flowLine_P 
 Max_Outer_flowLine_P: $Max_Outer_flowLine_P"

 /bin/echo "\n ** Pulmonary artery  Segmentation with Opfront:"	   # segmentation (as an input everything should be dicom)
 $Path_Opfront $Vol_dcm $Seg_Net/Auto.dcm $Seg_Pulmo_Org ${inner_smooth_P} ${grid_spacing_P} ${kernel_size_P} ${sample_interval_P} ${edge_length_P} ${regularization_P} ${Max_inner_flowLine_P} ${Max_Outer_flowLine_P} 
 /bin/echo Pulmonary artery  Segmentation Ends on: `date`

 ################################################################################################################
 ######## 3. Dice and surface distance statistics for original volume #########################
 /bin/echo  "**************************************************************************************************" 
 /bin/echo  "***** 3. Dice and surface distance statistics for original volume **************"  
 
 /bin/echo " 1.a) Cut both segmentations and calculate dice and jaccard coefficient: "
 module load mcr/R2015b
 $Matlab_path/Dice_Correct_Pulmo_ShuaiResults $Manual_LeftCenterline $Manual_RightCenterline $Manual_Segmentation $Seg_Pulmo_Org/${File}_Segmentation.dcm $Seg_Pulmo_Org/Manual $Seg_Pulmo_Org/Auto

 /bin/echo " 3.b) calculate the mean surface distance (mean, max, min, std of the surface distance): "
 $Path_MeanDIST $Seg_Pulmo_Org/Auto.mhd $Seg_Pulmo_Org/Manual.mhd $Seg_Pulmo_Org/MSD_CompleteVolume.txt
 rm $Seg_Pulmo_Org/*.mhd 
 rm $Seg_Pulmo_Org/*.raw

 /bin/echo Dice, Mean surface distance for the Pulmonary artery segmentation with original image Ends on: `date`

##############################################################################
############ 5. Reduce the contrast of the volume ############################
 /bin/echo "*****************************************************************" 
 /bin/echo "*********** 5. Reduce the contrast of the volume ****************" 
 
 module load mcr
 $Matlab_path/ContrastChanged $Vol_dcm $Seg_Pulmo_Contrast   
 module unload mcr 
 
 #####################################################################################################################
 ########### 5. Pulmonary artery  segmentation initialized with the deep learning results on Contrast reduced Volume #############
 /bin/echo  "********************************************************************************************************" 
 /bin/echo  "***** 5. Pulmonary artery segmentation initialized with the deep learning results Contrast reduced Volume ********"  
 #Parameters_Aorta:
 inner_smooth_P="50"  
 grid_spacing_P="0.8"  	
 kernel_size_P="31" 
 sample_interval_P="0.5" 
 edge_length_P="0.8"  
 regularization_P="1.6"  
 Max_inner_flowLine_P="14"  
 Max_Outer_flowLine_P="20"
 
 /bin/echo "Pulmonary artery  parameters are:
 inner_smooth_P: $inner_smooth_P
 grid_spacing_P: $grid_spacing_P    
 kernel_size_P: $kernel_size_P    
 sample_interval_P: $sample_interval_P 
 edge_length_P: $edge_length_P     
 regularization_P: $regularization_P   
 Max_inner_flowLine_P: $Max_inner_flowLine_P 
 Max_Outer_flowLine_P: $Max_Outer_flowLine_P"

 /bin/echo "\n ** Pulmonary artery  Segmentation with Opfront:"	   # segmentation (as an input everything should be dicom)
 $Path_Opfront $Seg_Pulmo_Contrast/${File}_Threshold150.dcm $Seg_Net/Auto.dcm $Seg_Pulmo_Contrast ${inner_smooth_P} ${grid_spacing_P} ${kernel_size_P} ${sample_interval_P} ${edge_length_P} ${regularization_P} ${Max_inner_flowLine_P} ${Max_Outer_flowLine_P} 
 /bin/echo Pulmonary artery  Segmentation Ends on: `date`

 ########################################################################################################################
 ######## 6. Dice and surface distance statistics for contrast reduced volume #########################
 /bin/echo  "***********************************************************************************************************" 
 /bin/echo  "***** 6. Dice, and surface distance statistics for contrast reduced volume ***************"  
 
 /bin/echo " 6.a) Cut both segmentations and calculate dice and jaccard coefficient: "
 module load mcr/R2015b
 $Matlab_path/Dice_Correct_Pulmo_ShuaiResults $Manual_LeftCenterline $Manual_RightCenterline $Manual_Segmentation $Seg_Pulmo_Contrast/${File}_Segmentation.dcm $Seg_Pulmo_Contrast/Manual $Seg_Pulmo_Contrast/Auto

 /bin/echo " 6.b) calculate the mean surface distance (mean, max, min, std of the surface distance): "
 $Path_MeanDIST $Seg_Pulmo_Contrast/Auto.mhd $Seg_Pulmo_Contrast/Manual.mhd $Seg_Pulmo_Contrast/MSD_CompleteVolume.txt
 rm $Seg_Pulmo_Contrast/*.mhd 
 rm $Seg_Pulmo_Contrast/*.raw
 
 /bin/echo Dice, Mean surface distance for the pulmonary artery segmentation with contrast reduced image Ends on: `date`

####################################################################################
 End_Total=$(date +%s)
 runtime_total=$((End_Total - Start_Total))
 runtime_total_print=$(python -c "print '%u:%02u' % ((${runtime_total})/60, (${runtime_total})%60)")
 /bin/echo Total Runtime : $runtime_total_print  
