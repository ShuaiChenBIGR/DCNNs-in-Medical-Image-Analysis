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
Manual_Segmentation=$3  #e.g. Aorta manual segmentation (dcm)   /vol4/vol4_Manual_3DMask.dcm
Manual_Centerline=$4  	#e.g. Aorta manual resampled centerline (txt) /vol4/Aorta/vol4_Aorta_Centerline_resampled.txt
Manual_Bif=$5			#e.g. Manual_Bif=/scratch/zsedghi/Data/DLCST_BifPoint/vol4_BifurcationPoint.txt
Output_folder=$6		#e.g. Output_folder=/scratch/zsedghi/Shuai_Results/Segmentation/vol4
      
if [ $# != "6" ]	 # countes the number of inputs and if less than 9 it gives error
then
	 echo "  "
	 echo " Not enough inputs."
	 echo "  "
	 echo "6 Inputs are required which should be in this order: 
	 1. Full volume (dcm) 
	 2.initialization for opfront (output of deep learning)
	 3.Manual Aorta Segmentation( dcm format)
	 4.Manual Aorta Centerline (resampled txt)
	 5.Manual pulmonary artery bifurcation point(txt file) 
	 6.Output folder  (a directory not a file name)
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
 Seg_Aorta_Org=$Output_folder/Aorta_Segmentation_OriginalVolume
 Seg_Aorta_Contrast=$Output_folder/Aorta_Segmentation_ContrastReduced
 Seg_Net=$Output_folder/Deeplearning_Accuracy

 #mkdir -p $Seg_Aorta_Org
 #mkdir -p $Seg_Aorta_Contrast
 mkdir -p $Seg_Net
 
#################################################################################################################################################
### 1. Cut manual and initial segmentation and get the initialization accuracy(DSC, MSD, DiameterProfile) All based on manual centerline ########
##########################################################################
 /bin/echo "********************************************************************************************************************************************" 
 /bin/echo "**** 1. Cut manual and initial segmentation and get the initialization accuracy(DSC, MSD, DiameterProfile) All based on manual centerline***" 
 /bin/echo "********************************************************************************************************************************************"  
 /bin/echo " 1.a) Cut both segmentations and calculate dice and jaccard coefficient: "
 /bin/echo " "
 module load mcr;
 $Matlab_path/Dice_Correct_ShuaiResults $Manual_Centerline $Manual_Segmentation $Auto_Initialization $Seg_Net/Manual $Seg_Net/Auto

 /bin/echo " 1.b) calculate the mean surface distance (mean, max, min, std of the surface distance): "
 $Path_MeanDIST $Seg_Net/Auto.mhd $Seg_Net/Manual.mhd $Seg_Net/MSD_CompleteVolume.txt
 rm $Seg_Net/*.mhd 
 rm $Seg_Net/*.raw
 
 /bin/echo " 1.c) calculate the diameter profile at different cross sectionals: "
 $Matlab_path/DiameterPerSlice_Shuai_Results $Manual_Segmentation $Seg_Net/Auto.dcm $Manual_Centerline $Manual_Centerline  $Manual_Bif $Seg_Net
 module unload mcr 

 /bin/echo Dice, Mean surface distance and diameter measurements for the initialaization Ends on: `date`

 # # ###################################################################################################################
 # # ########### 2. Aorta segmentation initialized with the deep learning results on original volume ###################
 # # /bin/echo  "********************************************************************************************************" 
 # # /bin/echo  "***** 2.  Aorta segmentation initialized with the deep learning results on original volume *************"  
 # # #Parameters_Aorta:
 # # inner_smooth_A="50"  
 # # grid_spacing_A="0.8"  	
 # # kernel_size_A="31" 
 # # sample_interval_A="0.5" 
 # # edge_length_A="0.8"  
 # # regularization_A="1.6"  
 # # Max_inner_flowLine_A="14"  
 # # Max_Outer_flowLine_A="20"
 
 # # /bin/echo "Aorta parameters are:
 # # inner_smooth_A: $inner_smooth_A
 # # grid_spacing_A: $grid_spacing_A    
 # # kernel_size_A: $kernel_size_A    
 # # sample_interval_A: $sample_interval_A 
 # # edge_length_A: $edge_length_A     
 # # regularization_A: $regularization_A   
 # # Max_inner_flowLine_A: $Max_inner_flowLine_A 
 # # Max_Outer_flowLine_A: $Max_Outer_flowLine_A"

 # # /bin/echo "\n ** Aorta Segmentation with Opfront:"	   # segmentation (as an input everything should be dicom)
 # # $Path_Opfront $Vol_dcm $Seg_Net/Auto.dcm $Seg_Aorta_Org ${inner_smooth_A} ${grid_spacing_A} ${kernel_size_A} ${sample_interval_A} ${edge_length_A} ${regularization_A} ${Max_inner_flowLine_A} ${Max_Outer_flowLine_A} 
 # # /bin/echo Aorta Segmentation Ends on: `date`

 # # ################################################################################################################
 # # ######## 3. Dice, Diameter profile and surface distance statistics for original volume #########################
 # # /bin/echo  "**************************************************************************************************" 
 # # /bin/echo  "***** 3. Dice, Diameter profile and surface distance statistics for original volume **************"  
 
 # # /bin/echo " 1.a) Cut both segmentations and calculate dice and jaccard coefficient: "
 # # module load mcr/R2015b
 # # $Matlab_path/Dice_Correct_ShuaiResults $Manual_Centerline $Manual_Segmentation $Seg_Aorta_Org/${File}_Segmentation.dcm $Seg_Aorta_Org/Manual $Seg_Aorta_Org/Auto

 # # /bin/echo " 3.b) calculate the mean surface distance (mean, max, min, std of the surface distance): "
 # # $Path_MeanDIST $Seg_Aorta_Org/Auto.mhd $Seg_Aorta_Org/Manual.mhd $Seg_Aorta_Org/MSD_CompleteVolume.txt
 # # rm $Seg_Aorta_Org/*.mhd 
 # # rm $Seg_Aorta_Org/*.raw
 
 # # /bin/echo " 3.b) calculate the diameter profile at different cross sectionals: "
 # # $Matlab_path/DiameterPerSlice_Shuai_Results $Manual_Segmentation $Seg_Aorta_Org/Auto.dcm $Manual_Centerline $Manual_Centerline $Manual_Bif $Seg_Aorta_Org
 # # module unload mcr 

 # # /bin/echo Dice, Mean surface distance and diameter measurements for the Aorta segmentation with original image Ends on: `date`

# # ##############################################################################
# # ############ 5. Reduce the contrast of the volume ############################
 # # /bin/echo "*****************************************************************" 
 # # /bin/echo "*********** 5. Reduce the contrast of the volume ****************" 
 
 # # module load mcr
 # # $Matlab_path/ContrastChanged $Vol_dcm $Seg_Aorta_Contrast   
 # # module unload mcr 
 
 # # #####################################################################################################################
 # # ########### 5. Aorta segmentation initialized with the deep learning results on Contrast reduced Volume #############
 # # /bin/echo  "********************************************************************************************************" 
 # # /bin/echo  "***** 5.  Aorta segmentation initialized with the deep learning results Contrast reduced Volume ********"  
 # # #Parameters_Aorta:
 # # inner_smooth_A="50"  
 # # grid_spacing_A="0.8"  	
 # # kernel_size_A="31" 
 # # sample_interval_A="0.5" 
 # # edge_length_A="0.8"  
 # # regularization_A="1.6"  
 # # Max_inner_flowLine_A="14"  
 # # Max_Outer_flowLine_A="20"
 
 # # /bin/echo "Aorta parameters are:
 # # inner_smooth_A: $inner_smooth_A
 # # grid_spacing_A: $grid_spacing_A    
 # # kernel_size_A: $kernel_size_A    
 # # sample_interval_A: $sample_interval_A 
 # # edge_length_A: $edge_length_A     
 # # regularization_A: $regularization_A   
 # # Max_inner_flowLine_A: $Max_inner_flowLine_A 
 # # Max_Outer_flowLine_A: $Max_Outer_flowLine_A"

 # # /bin/echo "\n ** Aorta Segmentation with Opfront:"	   # segmentation (as an input everything should be dicom)
 # # $Path_Opfront $Seg_Aorta_Contrast/${File}_Threshold150.dcm $Seg_Net/Auto.dcm $Seg_Aorta_Contrast ${inner_smooth_A} ${grid_spacing_A} ${kernel_size_A} ${sample_interval_A} ${edge_length_A} ${regularization_A} ${Max_inner_flowLine_A} ${Max_Outer_flowLine_A} 
 # # /bin/echo Aorta Segmentation Ends on: `date`

 # # ########################################################################################################################
 # # ######## 6. Dice, Diameter profile and surface distance statistics for contrast reduced volume #########################
 # # /bin/echo  "***********************************************************************************************************" 
 # # /bin/echo  "***** 6. Dice, Diameter profile and surface distance statistics for contrast reduced volume ***************"  
 
 # # /bin/echo " 6.a) Cut both segmentations and calculate dice and jaccard coefficient: "
 # # module load mcr/R2015b
 # # $Matlab_path/Dice_Correct_ShuaiResults $Manual_Centerline $Manual_Segmentation $Seg_Aorta_Contrast/${File}_Segmentation.dcm $Seg_Aorta_Contrast/Manual $Seg_Aorta_Contrast/Auto

 # # /bin/echo " 6.b) calculate the mean surface distance (mean, max, min, std of the surface distance): "
 # # $Path_MeanDIST $Seg_Aorta_Contrast/Auto.mhd $Seg_Aorta_Contrast/Manual.mhd $Seg_Aorta_Contrast/MSD_CompleteVolume.txt
 # # rm $Seg_Aorta_Contrast/*.mhd 
 # # rm $Seg_Aorta_Contrast/*.raw
 
 # # /bin/echo " 6.b) calculate the diameter profile at different cross sectionals: " 
 # # $Matlab_path/DiameterPerSlice_Shuai_Results $Manual_Segmentation $Seg_Aorta_Contrast/Auto.dcm $Manual_Centerline $Manual_Centerline $Manual_Bif $Seg_Aorta_Contrast

 # # module unload mcr 

 # # /bin/echo Dice, Mean surface distance and diameter measurements for the Aorta segmentation with contrast reduced image Ends on: `date`

# # ####################################################################################
 # # End_Total=$(date +%s)
 # # runtime_total=$((End_Total - Start_Total))
 # # runtime_total_print=$(python -c "print '%u:%02u' % ((${runtime_total})/60, (${runtime_total})%60)")
 /bin/echo Total Runtime : $runtime_total_print  
