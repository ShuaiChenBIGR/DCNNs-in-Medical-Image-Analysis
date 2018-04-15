function Dice_Correct_Pulmo_ShuaiResults(ManLeftCenterline,ManRightCenterline, ManualSegentation, AutoSegmentation, OutputManualName,OutputAutoName)
% This function cuts both manual and automatic segmentations from the
% first and last points of the centerline in axial slice. the mhd and dcm format of both corped segmentations
% and the dice and jaccard of them is saved at the output directory. For
% the output only give the directory and the name without the extention.
% 
% Inputs:
%       ManualCenterline = the txt file with a centerline point per mm(z
%       pixelspacing). It can be in both voxel or world coordinate
%       ManualSegentation = the manual volume of a non-contrast CT. it can
%       be either in dicom or in mhd format
%       AutoSegmentation = the automatic volume of a non-contrast CT. it can
%       be either in dicom or in mhd format
%       OutputManualName= the directory and the name of the corrected
%       manual segmentation, with no extention becouse the output will be saved in
%       both mhd and dcm formats
%       OutputAutoName= the directory and the name of the corrected
%       Automatic segmentation, with no extention becouse the output will be saved in
%       both mhd and dcm formats
% Outputs:
%       Dice= a txt file only saving the dice of the corrected
%       segmentations. in name Dice.txt
%       JaccardCoefficient= a txt file only saving the jaccard Index of the corrected
%       segmentations. in name jaccard.txt
%       Auto.mhd  &  Auto.dcm = The corrected automatic segmentation which
%       starts from Discending Aorta (first Centerline poin)t and ends at
%       Ascending Aorta (last centerline point)
%       Manual.mhd  &  Manual.dcm = The corrected Manual segmentation which
%       starts from Discending Aorta (first Centerline poin)t and ends at
%       Ascending Aorta (last centerline point)
% 
% EX:
%  ManLeftCenterline='F:\Project-Zahra\Data\DLCST_Manual_Centerlines_Seeds\vol57\Pulmonary_Left\vol57_Pulmonary_Left_Centerline_resampled.txt';
%  ManRightCenterline='F:\Project-Zahra\Data\DLCST_Manual_Centerlines_Seeds\vol57\Pulmonary_Right\vol57_Pulmonary_Right_Centerline_resampled.txt';
%  ManualSegentation='F:\Project-Zahra\Data\DLCST_Manual_Centerlines_Seeds\vol57\Full_Pulmonary\vol57_Manual_Connected_Mask.dcm';
%  AutoSegmentation='D:\Shuai-Deeplearning_Aorta\results_20180309\vol57\masksPulPredicted.dcm';
%  OutputManualName='F:\Results\Test\man';
%  OutputAutoName='F:\Results\Test\auto';
%  Dice_Correct_ShuaiResults(ManLeftCenterline,ManRightCenterline, ManualSegentation, AutoSegmentation, OutputManualName,OutputAutoName)
% 
% Zahra Sedghi
%  17 Dec 2017
%% Check errors
tic
if (nargin ~= 6)
    error ('myApp:argChk',...
        'Not enough inputs. \n input1:  Manual leftCenterline (txt),\n input2 Manual Right Centerline (txt),\n input3:  Manual Segmentation (dcm/mhd), \n input4: Automatic Segmentation (dcm/mhd) \n input5: Output directory and name for manual image(no extention) \n input6: Output directory and name for Automatic image(no extention) \n')
end

%% load the manual and automatic volumes and the manual centerline centerline
% in here both manual and auto segmentations are the same becouse I am comparing them with manual centerlines
[manual_info,ManualSeg,ManleftCenter,ManRightCenter,auto_info,AutoSeg,...
    exten_Man, exten_Auto]= load_PulmoImg_Center_MHDorDCM(ManualSegentation,...
    ManLeftCenterline,ManRightCenterline,AutoSegmentation);

%% Cut the volumes based on the manual Centerline at Axial slices
% get the firat and las centerline p[oint
PR = ManleftCenter(1,:); % start of centerline is from pulmonary root
LP = ManleftCenter(end,:); % end is before second pulmonary bifurcation
RP = ManRightCenter(end,:);

for i=1:size(ManualSeg,3)
    if i<PR(3)
        AutoSeg(:,:,i) = 0;
        ManualSeg(:,:,i) = 0;
    end
end

pl=[LP(1)+10,LP(2)-8,LP(3)];  % fit a lne to 3 points of left pulmonary to cut it in an order prependicular to its centerline (the points are selected by vision inspections)
ml=(pl(2)-LP(2))/(pl(1)-LP(1));
for i=1:size(ManualSeg,3)
    for k =1:size(ManualSeg,1)
        for j=1:size(ManualSeg,2)
           if  ( k >= ml*(j-LP(1))+LP(2))
              ManualSeg(k,j,i)=0;
              AutoSeg(k,j,i) = 0;
           end
           if((k<RP(1)))
            AutoSeg(j,k,i) = 0;
            ManualSeg(j,k,i) = 0;
          end
        end
    end
end


% view2D(ManualSeg,1)
% view2D(AutoSeg,1)
%% convert to binaries with min=0, max=255 (make binaries with 0 and 255 intensities)
auto_min = min(min(min(AutoSeg)));
auto_max = max(max(max(AutoSeg)));
manual_min = min(min(min(ManualSeg)));
manual_max = max(max(max(ManualSeg)));

if auto_max < 255
    AutoSeg(AutoSeg~=0)=255;
end
if manual_max < 255
    ManualSeg(ManualSeg~=0)=255;
end

%% compute Dice overlap (Dice similarity coefficient)
[RootFolder ,name,ext] = fileparts(OutputManualName);
Dice = 2*(nnz(ManualSeg & AutoSeg))/(nnz(AutoSeg)+nnz(ManualSeg));  %n = nnz(X) returns the number of nonzero elements in matrix X.
[jaccardIdx,jaccardDist] = jaccard_coefficient(ManualSeg,AutoSeg);
savepath=fullfile(RootFolder,'Dice.txt');
saveJcard=fullfile(RootFolder,'jaccard.txt');
dlmwrite(savepath,Dice);
dlmwrite(saveJcard,jaccardIdx);

%% save as meta data images
% [RootFolder ,name,ext] = fileparts(Output);
% DirManual   =  fullfile(RootFolder,'Manual.mhd');
% DirAuto     =  fullfile(RootFolder,'Auto.mhd');
% DirMan_DCM  =  fullfile(RootFolder,'Manual.dcm');
% DirAuto_DCM =  fullfile(RootFolder,'Auto.dcm');

DirManual   =  strjoin({OutputManualName,'.mhd'},'');
DirAuto     =  strjoin({OutputAutoName,'.mhd'},'');
DirMan_DCM  =  strjoin({OutputManualName,'.dcm'},'');
DirAuto_DCM =  strjoin({OutputAutoName,'.dcm'},'');

if  isequal(exten_Man ,'.dcm')
    write2mhd(ManualSeg, manual_info, DirManual);  % write in mhd format
    ManualSeg_4D(:,:,1,:)=ManualSeg(:,:,:);        % write in dcm format
    dicomwrite(ManualSeg_4D,DirMan_DCM, manual_info,'CreateMode', 'copy');
    
elseif isequal(exten_Man ,'.mhd')
    ElementM=manual_info.ElementSpacing;
    TransformM=manual_info.TransformMatrix;
    OffsetM=manual_info.Offset;
    AnatomicalOrientationM=manual_info.AnatomicalOrientation;
    CenterOfRotationM=manual_info.CenterOfRotation; 
    metaImageWrite(ManualSeg, DirManual,'ElementSpacing', ElementM,'CompressedData','False','Offset', ...
    OffsetM, 'TransformMatrix',TransformM,'CenterOfRotation' , CenterOfRotationM ,'AnatomicalOrientation', AnatomicalOrientationM )
end

if isequal(exten_Auto ,'.dcm')
    write2mhd(AutoSeg, auto_info , DirAuto);    % write in mhd format
    AutoSeg_4D(:,:,1,:)=AutoSeg(:,:,:);       % write in dcm format
    dicomwrite(AutoSeg_4D,DirAuto_DCM, auto_info,'CreateMode', 'copy');
    
elseif isequal(exten_Auto ,'.mhd')
    ElementA=auto_info.ElementSpacing;
    TransformA=auto_info.TransformMatrix;
    OffsetA=auto_info.Offset;
    AnatomicalOrientationA=auto_info.AnatomicalOrientation;
    CenterOfRotationA=auto_info.CenterOfRotation; 
    metaImageWrite(AutoSeg, DirAuto,'ElementSpacing', ElementA,'CompressedData','False','Offset', ...
    OffsetA, 'TransformMatrix',TransformA,'CenterOfRotation' , CenterOfRotationA ,'AnatomicalOrientation', AnatomicalOrientationA )
end

% annoncing the user that it hase been saved
fprintf('\n The Dice Similarity Coefficienti DSC =  "%0.5f"  and the jaccard = "%0.5f" \n  and the croped manual and automatic images are saved at: \n "%s"  \n \n',Dice,jaccardIdx,RootFolder);

end
% to compile on the cluster
%   mcc -mv -R singleCompThread -R nojvm -R nodesktop -R nosplash Dice_Correct_Pulmo_ShuaiResults.m


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% loading image and centerlines Function %%%%%%%%%%%%%%%%%
function [manual_info,ManualSeg,ManleftCenter,ManRightCenter,auto_info,AutoSeg,exten_Man, exten_Auto]= load_PulmoImg_Center_MHDorDCM (ManualSegentation,ManualleftCenter,ManualRightCenter,AutoSegmentation)
ManleftCenter  =dlmread(ManualleftCenter);
ManRightCenter  =dlmread(ManualRightCenter);

[pathM,nameM, exten_Man]=fileparts(ManualSegentation);
if isequal(exten_Man ,'.mhd')    % if the loaded image is in mhd format
    manual_info  =  metaImageInfo(ManualSegentation);
    ManualSeg   =  metaImageRead(manual_info);
    if (abs(ManleftCenter(1,3))>size(ManualSeg,3)|| any(ManleftCenter(:)<0))  % if the coordinate is in world coordinate convert it to voxel
        ManleftCenter = convWorld2VoxelMHD(ManleftCenter,manual_info);
        ManRightCenter = convWorld2VoxelMHD(ManRightCenter,manual_info);
    else
        ManleftCenter = convWorld2Voxel(ManleftCenter,manual_info); 
        ManRightCenter = convWorld2Voxel(ManRightCenter,manual_info); 
    end
elseif isequal(exten_Man ,'.dcm')  % if the loaded image is in dcm format
    ManualSeg   = squeeze(dicomread(ManualSegentation));
    manual_info  = dicominfo(ManualSegentation);
    if (abs(ManleftCenter(1,3))>size(ManualSeg,3)|| any(ManleftCenter(:)<0))
        ManleftCenter = convWorld2Voxel(ManleftCenter,manual_info);
        ManRightCenter = convWorld2Voxel(ManRightCenter,manual_info);
    else
        ManleftCenter = convWorld2Voxel(ManleftCenter,manual_info);
        ManRightCenter = convWorld2VoxelMHD(ManRightCenter,manual_info);
    end
end

[pathA,nameA, exten_Auto]=fileparts(AutoSegmentation);
if isequal(exten_Auto ,'.mhd')    % if the loaded image is in mhd format
    auto_info     =  metaImageInfo(AutoSegmentation);
    AutoSeg =  metaImageRead(manual_info);
   elseif isequal(exten_Auto ,'.dcm')    % if the loaded image is in dcm format
    AutoSeg = squeeze(dicomread(AutoSegmentation));
    auto_info     = dicominfo(AutoSegmentation);
    end
end

%%%%%%%%%%%%%%%%%% Saving a dicom image as mhd Function %%%%%%%%%%%%%%%%%%%
function write2mhd(imge_matrix, info , OutputMHD )
Spacing     =  [info.PixelSpacing(1,1), info.PixelSpacing(2,1), info.SpacingBetweenSlices];
type_dcm    =  class(imge_matrix);
NDim        =  info.FormatVersion;
Offset      =  info.ImagePositionPatient;

switch type_dcm
    case 'uint8'
        type = 'MET_UCHAR';
    case 'int8'
        type = 'MET_CHAR';
    case 'uint16'
        type = 'MET_USHORT';
    case 'int16'
        type = 'MET_SHORT';
    case 'uint32'
        type = 'MET_UINT';
    case 'int32'
        type = 'MET_INT';
    case 'single'
        type = 'MET_FLOAT';
    case 'double'
        type = 'MET_DOUBLE';
end

metaImageWrite(imge_matrix, OutputMHD, 'ElementType', type, 'ElementSpacing', Spacing,... 
    'NDims',NDim,'CompressedData','False','Offset',Offset)
end
