#!/bin/bash
# print the command lines
# set -x
# Call the full automatic Aorta segmentation by using regestration for initialization and mask for medialness 29-09-2017

Que="day"
Memory="15G"
Input_Folder=/scratch/zsedghi/Data/DLCST
Output_Folder=/scratch/schen/Evaluation/Final_Accuracy
Manual_Folder=/scratch/zsedghi/Data/DLCST_Manual_Centerlines_Seeds
Manual_Bif_Folder=/scratch/zsedghi/Data/DLCST_BifPoint
Path_Code=/scratch/schen/Evaluation/Scripts/Segment_Shuai_Results.sh
Initialization=/scratch/schen/Evaluation/Results

# Create output and log folders
Log_Folder=$Output_Folder/Logs
NOW=$(date +%d-%m-%Y)
Log_File=$Log_Folder/Call_AutoSeedPint_Centerline_${NOW}.log 
mkdir -p $Output_Folder
mkdir -p $Log_Folder

# Write parameters in log file
echo "  " > $Log_File #for having a new line
echo "Segmentation of Aofta by initializing with deep learning results from Shuai's code 19-12-2017" >> $Log_File
echo "Input Data Folder: ${Input_Folder}" >> $Log_File
echo -e "Manual Folder: ${Manual_Folder}\n" >> $Log_File
echo -e "Output Folder: ${Output_Folder}\n" >> $Log_File
echo -e "Code path: ${Path_Code}\n" >> $Log_File
echo "$(date +%d-%m-%Y----%T)" >> $Log_File
eval cat $Log_File

# Call Volumes in Input Folder
names="57 58 69 73 85 93 167 221 371 658 754 809 2157 3418 3885"
for ii in ${names} 
do
for vol in ${Input_Folder}/vol${ii}.dcm
do
	 File_base=$(basename "${vol}" .dcm)
	 File=${File_base%%_*} 
	 Output=$Output_Folder/${File}
	 mkdir -p $Output
	 
	 Initial=$Initialization/${File}/DICOM/masksAortaPredicted.dcm
	 Manual_Seg=$Manual_Folder/${File}/Aorta/${File}_Manual_3DMask.dcm
	 Manual_Center=$Manual_Folder/${File}/Aorta/${File}_Aorta_Centerline_resampled.txt
	 Manual_Bif=$Manual_Bif_Folder/${File}_BifurcationPoint.txt
	 
	 Job_name="AutoSeg_${File}"
	 Call="$Path_Code $vol $Initial $Manual_Seg $Manual_Center $Manual_Bif $Output "
	 echo ${Call} | qsub -q $Que -l h_vmem=${Memory} -j y -N ${Job_name} -V -o $Log_Folder/${Job_name}_${NOW}.log
	 echo "$Call" >> $Log_File

done
done

