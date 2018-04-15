function Dice_Correct_ShuaiResults(ManualCenterline, ManualSegentation, AutoSegmentation, OutputManualName,OutputAutoName)
% This function cuts both manual and automatic segmentations from the
% first and last points of the centerline in axial slice. the mhd and dcm format of both corped segmentations
% and the dice and jaccard of them is saved at the output directory. For
% the output only give the directory and the nme without the extention.
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
%  ManualCenterline='F:\Project-Zahra\Data\DLCST_Manual_Centerlines_Seeds\vol6\Aorta\vol6_Aorta_Centerline_resampled.txt';
%  ManualSegentation='F:\Project-Zahra\Data\DLCST_Manual_Centerlines_Seeds\vol6\Aorta\vol6_Manual_3DMask.dcm';
%  AutoSegmentation='D:\Shuai-Deeplearning_Aorta\vol6\masksTestPredicted.dcm';
%  OutputManualName='F:\Results\Test\man';
%  OutputAutoName='F:\Results\Test\auto';
%  Dice_Correct_ShuaiResults(ManualCenterline, ManualSegentation, AutoSegmentation, OutputManualName,OutputAutoName)
% 
% Zahra Sedghi
%  17 Dec 2017
%% Check errors
tic
if (nargin ~= 5)
    error ('myApp:argChk',...
        'Not enough inputs. \n input1:  Manual Centerline (dcm/mhd),\n input2:  Manual Segmentation (dcm/mhd), \n input3: Automatic Segmentation (dcm/mhd) \n input4: Output directory and name for manual image(no extention) \n input5: Output directory and name for Automatic image(no extention) \n')
end

%% load the manual and automatic volumes and the manual centerline centerline
[manual_info,ManualSeg,ManCenter,auto_info,AutoSeg,AutoCenter,...
    exten_Man, exten_Auto]= load_Img_MHDorDCM (ManualSegentation,ManualCenterline,AutoSegmentation,ManualCenterline); 
% in here both manual and auto segmentations are the same becouse I am comparing them with manual centerlines

%% Cut the volumes based on the manual Centerline at Axial slices
% get the firat and las centerline p[oint
DA = ManCenter(1,:); % centerline starts from desending aorta to acsending aorta so I take only the are between the start and end points
AA = ManCenter(end,:); 

if sqrt(sum((DA(1:2)-AA(1:2)).^2))>85
    RadiusRange=62;  % In a region with radius 62 everthing below the ascending aorta is set to zero
else
    RadiusRange=35;  % In a region with radius 35 everthing below the ascending aorta is set to zero
end

for i=1:size(ManualSeg,3)
    if i<DA(3)
        AutoSeg(:,:,i) = 0;
        ManualSeg(:,:,i) = 0;
    end
    if i<AA(3)
        for j=1:size(ManualSeg,1)
            for k=1:size(ManualSeg,2)
                if (((j-AA(2))^2 + (k-AA(1))^2 < RadiusRange^2)|| j<AA(2)+15)
                    AutoSeg(j,k,i) = 0;
                    ManualSeg(j,k,i) = 0;
                end
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
%   mcc -mv -R singleCompThread -R nojvm -R nodesktop -R nosplash Dice_Correct_ShuaiResults.m


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% loading image and centerlines Function %%%%%%%%%%%%%%%%%
function [manual_info,img_manual,ManCenter,auto_info,img_automatic,AutoCenter,exten_Man, exten_Auto]= load_Img_MHDorDCM (ManualSegentation,ManualCenterline,AutoSegmentation,AutoCenterline)
ManCenter  =dlmread(ManualCenterline);
AutoCenter = dlmread(AutoCenterline);

[pathM,nameM, exten_Man]=fileparts(ManualSegentation);
if isequal(exten_Man ,'.mhd')    % if the loaded image is in mhd format
    manual_info  =  metaImageInfo(ManualSegentation);
    img_manual   =  metaImageRead(manual_info);
    if (abs(ManCenter(1,3))>size(img_manual,3)|| any(ManCenter(:)<0))  % if the coordinate is in world coordinate convert it to voxel
        ManCenter = convWorld2VoxelMHD(ManCenter,manual_info);
    else
        ManCenter = convWorld2Voxel(ManCenter,manual_info); 
    end
elseif isequal(exten_Man ,'.dcm')  % if the loaded image is in dcm format
    img_manual   = squeeze(dicomread(ManualSegentation));
    manual_info  = dicominfo(ManualSegentation);
    if (abs(ManCenter(1,3))>size(img_manual,3)|| any(ManCenter(:)<0))
        ManCenter = convWorld2Voxel(ManCenter,manual_info);
    else
         ManCenter = convWorld2VoxelMHD(ManCenter,manual_info);
    end
end

[pathA,nameA, exten_Auto]=fileparts(AutoSegmentation);
if isequal(exten_Auto ,'.mhd')    % if the loaded image is in mhd format
    auto_info     =  metaImageInfo(AutoSegmentation);
    img_automatic =  metaImageRead(manual_info);
    if (abs(AutoCenter(1,3))>size(img_automatic,3) || any(AutoCenter(:)<0)) % if the coordinate is in world coordinate convert it to voxel
        AutoCenter = convWorld2VoxelMHD(AutoCenter,auto_info);
    else
        AutoCenter = convWorld2Voxel(AutoCenter,auto_info);
    end
elseif isequal(exten_Auto ,'.dcm')    % if the loaded image is in dcm format
    img_automatic = squeeze(dicomread(AutoSegmentation));
    auto_info     = dicominfo(AutoSegmentation);
    if (abs(AutoCenter(1,3))>size(img_automatic,3) || any(AutoCenter(:)<0))
        AutoCenter = convWorld2Voxel(AutoCenter,auto_info);
    else
        AutoCenter = convWorld2VoxelMHD(AutoCenter,auto_info); 
    end
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
