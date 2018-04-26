function [DicCompvol,JaccardComp,Dices,SD,DiamMapMan,DiamMapAuto,Error]=LoadDiameterStatisticsPerSlice_Max_Av(DiamProfilePath,DiceSurfDistPath)
%this file loads the diameter and eror profile and dice and surface
%distance for statistics per slice
% %  EX:
%     DiamProfilePath    = fullfile(SubFolder,'DiameterProfile.mat');
%     DiceSurfDistPath   = fullfile(SubFolder,'StatisticDiceSurfaceDistancce.mat');
%
% Zahra Sedghi

%% create the output matrices:
    Slices=[-3:3, 3:-1:-2];  % slices location compared to BifLevel at DA to AA
    Dices(:,1)=Slices;
    SD(:,1)=Slices;
    DiamMapMan(:,1)=Slices;
    DiamMapAuto(:,1)=Slices;
    Error(:,1) =Slices;

    % load mat files:
    if isequal(exist(DiamProfilePath,'file'),2)
         load (DiamProfilePath);
    else Error(:,2:5)=NaN; DiamMapAuto(:,2:8)=NaN;  DiamMapMan(:,2:8)=NaN;  end
    if isequal(exist(DiceSurfDistPath,'file'),2)
         load (DiceSurfDistPath); 
     if isequal(exist(DiceSurfDistPath,'file'),2)
         if exist ('DicCompelete')
            DicCompvol=DicCompelete;
         elseif exist ('Dice')
             DicCompvol=Dice;
         end
     end
      if exist ('jaccard') 
          JaccardComp=jaccard;
      else JaccardComp=NaN; end
    else DicCompvol=NaN;Dices(:,2)=NaN;SD(:,2:3)=NaN;JaccardComp=NaN;end
     
    if exist('DicePerSlice'); Dices(:,2)=DicePerSlice; else Dices(:,2)=NaN; end
    if exist('SurfaceDistancePerSlice'); SD(:,2:3)=SurfaceDistancePerSlice(:,1:2); else SD(:,2:3)=NaN; end
    if exist('ErrorProfileBifSlicesSeperateCenterline'); Error(:,2:5)=ErrorProfileBifSlicesSeperateCenterline(:,1:4); else Error(:,2:5)=NaN; end
    if exist('AutomaticDiameterBifSlicesMap'); DiamMapAuto(:,2:8)=AutomaticDiameterBifSlicesMap(:,1:7); else DiamMapAuto(:,1:8)=NaN; end
    if exist('ManualDiameterBifSlicesMap'); DiamMapMan(:,2:8)=ManualDiameterBifSlicesMap(:,1:7); else DiamMapMan(:,2:8)=NaN; end
 