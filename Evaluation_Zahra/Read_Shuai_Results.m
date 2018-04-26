% reade the dice and surface distance and diameter profile from DLCST data
% This folder gets all the statistics required about diameter and surface 
% distances in slices prependecular to the centerline in 26 DLCST data for Shuai results. 
% the data are created by script: "Dice_Correct_ShuaiResults.m" for aorta
% and "Dice_Correct_Pulmo_ShuaiResults.m" for pulmonay results. Surface
% distance is done on cluster by itktools. (average mean surface distance)
% 
% change the input folder and saving folder anytime you want to run the
% code:
%       Root_Folder= where the results are saved  
%       save_Root = where you ant the results to be saved
% 
% Zahra Sedghi
%  17 Dec 2017
%% clear the workspace
clc
clear all
close all

%% load folders
Root_Folder='/hdd2/PythonCodes/Aorta_Segmentation_2D_3D_Shuai/testingSet/Results';
% Root_Folder='D:\Shuai-Deeplearning_Aorta\Segmentation';
save_Root=fullfile(Root_Folder,'Deeplearning_Accuracy');  % this saves the results in the same directory in a folder named Deeplearning_Accuracy
% save_Root=fullfile(Root_Folder,'Aorta_Segmentation_OriginalVolume');
% save_Root=fullfile(Root_Folder,'Aorta_Segmentation_ContrastReduced');
mkdir (save_Root)

%%  reading the name of the folders and saving them in a cell file 
tic
folder_list = dir(Root_Folder); % list of folders to process 
isub = [folder_list(:).isdir];     % to search for the ones which are only directories
Folders = {folder_list(isub).name}'; % copy names of ONLY directories
Folders(ismember(Folders,{'.','..'})) = []; % remove '.' and '..'
numFolder = length(Folders);

%% get the subfolders and the including mat files
 K=1;
for ii = 1:numFolder
     if ismember(Folders{ii}, {'logs','Logs','Deeplearning_Accuracy','Aorta_Segmentation_OriginalVolume','Aorta_Segmentation_ContrastReduced'})  % here you can exclude any folder that is not a result folder
         continue
     end  

    SubFolder = fullfile(Root_Folder, Folders{ii},'Deeplearning_Accuracy');
%     SubFolder = fullfile(Root_Folder, Folders{ii},'Aorta_Segmentation_OriginalVolume');
%     SubFolder = fullfile(Root_Folder, Folders{ii},'Aorta_Segmentation_ContrastReduced');
    
    % get the volume name
    row=K;
    volname=str2double(regexp( Folders{ii},'\d*','match')) 
    Variable_list(row,1) = volname;
    Shortcut_list(row,1)= volname;
    slicesNum=13;
   
    % load the mat and txt files for dice surface distance and diameter statistics   
    DiamProfilePath    = fullfile(SubFolder,'DiameterProfile.mat');
    DiceSurfDistPath   = fullfile(SubFolder,'StatisticDiceSurfaceDistancce.mat');
    CompSurfDistPath   = fullfile(SubFolder,'MSD_CompleteVolume.txt');
    
    [DicCompvol,JaccardComp,Dices,SD,DiamMapMan,DiamMapAuto,Error]=LoadDiameterStatisticsPerSlice_Max_Av(DiamProfilePath,DiceSurfDistPath);
    Dice_Stats{row,1}=volname; Dice_Stats{row,2}=Dices;
    SD_Stats{row,1}=volname; SD_Stats{row,2}=SD;
    DiamMapMan_Stats{row,1}=volname; DiamMapMan_Stats{row,2}=DiamMapMan;
    DiamMapAuto_Stats{row,1}=volname; DiamMapAuto_Stats{row,2}=DiamMapAuto;
    Error_Stats{row,1}=volname; Error_Stats{row,2}=Error;
    
    % create the matrices starting with volume names:
    DicePerSlice(row,1:slicesNum+1)=[volname,Dices(:,2)'];
    MaxSDPerSlice(row,1:slicesNum+1)=[volname,SD(:,3)'];
    MeanSDPerSlice(row,1:slicesNum+1)=[volname,SD(:,2)'];
    DiamMapAuto_Av(row,1:slicesNum+1)=[volname,DiamMapAuto(:,3)'];
    DiamMapAuto_Max(row,1:slicesNum+1)=[volname,DiamMapAuto(:,4)'];
    AreaMapAuto(row,1:slicesNum+1)=[volname,DiamMapAuto(:,2)'];
    DiamMapMan_Av(row,1:slicesNum+1)=[volname,DiamMapMan(:,3)'];
    DiamMapMan_Max(row,1:slicesNum+1)=[volname,DiamMapMan(:,4)'];
    AreaMapMan(row,1:slicesNum+1)=[volname,DiamMapMan(:,2)'];
    ErrorArea(row,1:slicesNum+1)=[volname,abs(Error(:,2)')];
    ErrorAvDiam(row,1:slicesNum+1)=[volname,abs(Error(:,3)')];
    ErrorMaxDiam(row,1:slicesNum+1)=[volname,abs(Error(:,4)')];
    ErrorMinDiam(row,1:slicesNum+1)=[volname,abs(Error(:,5)')];

    % get the complete volume dice & jcaard & SD
    DicePath   = fullfile(SubFolder,'Dice.txt');
    if isequal(exist(DicePath,'file'),2) 
         Variable_list(row,2)=dlmread(DicePath); 
    else
         Variable_list(row,2)=NaN;
    end
    
    JaccardPath   = fullfile(SubFolder,'jaccard.txt');
    if isequal(exist(JaccardPath,'file'),2) 
         Variable_list(row,6)=dlmread(JaccardPath); 
    else
         Variable_list(row,6)=NaN;
    end

    if isequal(exist(CompSurfDistPath,'file'),2) 
        statistic=GetSurfaceDistanceStatistics(CompSurfDistPath); % gives non existing MSD as NaN
        Variable_list(row,3:5)=statistic(2:end);
    else
        Variable_list(row,3:5)=[NaN,NaN,NaN];
    end
    
       % clear the variables for no missing up
     varlist = {'DicCompvol', 'JaccardComp','Dices',...
         'SD','DiamMapMan','DiamMapAuto','Error'};

     clear(varlist{:})

K = K+1;
end
toc
 
%% convert Nan to zero
% Variable_list(all(isnan(Variable_list(:,2:end)),2),2:end)=0;
%% save the variables:
  
save(fullfile(save_Root,'Dice_Stats.mat'),'Dice_Stats')
save(fullfile(save_Root,'SD_Stats.mat'),'SD_Stats')
save(fullfile(save_Root,'DiamMapMan_Stats.mat'),'DiamMapMan_Stats')
save(fullfile(save_Root,'DiamMapAuto_Stats.mat'),'DiamMapAuto_Stats')
save(fullfile(save_Root,'Error_Stats.mat'),'Error_Stats')

save(fullfile(save_Root,'DicePerSlice.mat'),'DicePerSlice')
save(fullfile(save_Root,'MaxSDPerSlice.mat'),'MaxSDPerSlice')
save(fullfile(save_Root,'MeanSDPerSlice.mat'),'MeanSDPerSlice')
save(fullfile(save_Root,'DiamMapAuto_Av.mat'),'DiamMapAuto_Av')
save(fullfile(save_Root,'DiamMapAuto_Max.mat'),'DiamMapAuto_Max')
save(fullfile(save_Root,'AreaMapAuto.mat'),'AreaMapAuto')
save(fullfile(save_Root,'DiamMapMan_Av.mat'),'DiamMapMan_Av')
save(fullfile(save_Root,'DiamMapMan_Max.mat'),'DiamMapMan_Max')
save(fullfile(save_Root,'AreaMapMan.mat'),'AreaMapMan')
save(fullfile(save_Root,'ErrorArea.mat'),'ErrorArea')
save(fullfile(save_Root,'ErrorAvDiam.mat'),'ErrorAvDiam')
save(fullfile(save_Root,'ErrorMaxDiam.mat'),'ErrorMaxDiam')
save(fullfile(save_Root,'ErrorMinDiam.mat'),'ErrorMinDiam')

save(fullfile(save_Root,'Variable_list.mat'),'Variable_list')

%% get ICC for all slices for average diameter
% slices are -3cm DA:3cmDA,3cm AA: -2cm AA
ManualDiameters=DiamMapMan_Av(:,2:end);
AutomaticDiameters=DiamMapAuto_Av(:,2:end);

 for slice= 1:slicesNum
     diameters=[ManualDiameters(:,slice),AutomaticDiameters(:,slice)];
     [Icc, LB, UB, F, df1, df2, pvalue] = ICC_v2(diameters, 'A-1' ); 
     IccPerslice_AvDiam(:,slice) = [Icc; LB; UB; pvalue]; 
     r = corrcoef(ManualDiameters(:,slice),AutomaticDiameters(:,slice));
     rPearson(1,slice)=r(1,2);
     r2Pearson(1,slice)=rPearson(1,slice).^2;
 end
ICCPerasonPerslice_AvDaiam=[IccPerslice_AvDiam;rPearson;r2Pearson];
save(fullfile(save_Root,'IccPerslice_AvDiam.mat'),'IccPerslice_AvDiam')
% save(fullfile(save_Root,'ICCPerasonPerslice_AvDaiam.mat'),'ICCPerasonPerslice_AvDaiam')

DiametersAll=[reshape(ManualDiameters, size(ManualDiameters,1)*size(ManualDiameters,2), 1),...
    reshape(AutomaticDiameters,size(AutomaticDiameters,1)*size(AutomaticDiameters,2), 1)];
[Icc, LB, UB, F, df1, df2, pvalue] = ICC_v2(DiametersAll, 'A-1' ); 
TotalICC_AvDiam=Icc;
save(fullfile(save_Root,'IccTotal_AvDiam.mat'),'TotalICC_AvDiam')
figure; scatter(DiametersAll(:,1),DiametersAll(:,2),'filled')
ylabel('Automatic Diameter(mm)','FontSize',13)
xlabel('Manual Diameter(mm)','FontSize',13)
title({[' Total ICC for All Measures Av Diam= ', num2str(TotalICC_AvDiam,'%0.2f')]},'FontSize',13)
lsline ; %Add least-squares line to scatter plot
set(gca, 'FontSize',13,'linew',1.1) % to set axes width and increase font size
axis equal
Name_Tfig=fullfile(save_Root,'IccTotal.jpg');
fig=gcf;
fig.PaperPositionMode = 'auto';
print('-dpng','-r0', Name_Tfig);    

figure('Position',[60 60 1600 820]);
nn=1;
for ii=1:7
    subplot(2,4,ii); 
    scatter(ManualDiameters(:,ii),AutomaticDiameters(:,ii),'filled')
    DAposition=-4+ii;
    ylabel('Av Automatic Diameter(mm)','FontSize',13)
    xlabel('Manual Diameter(mm)','FontSize',13)
    title({[num2str(DAposition),' cm DA, ICC = ', num2str(IccPerslice_AvDiam(1,ii),'%0.2f')]},'FontSize',13)
    lsline ; %Add least-squares line to scatter plot
    set(gca, 'FontSize',13,'linew',1.1) % to set axes width and increase font size
    axis equal
       
end
Name_DA=fullfile(save_Root,'IccScatterAvDaim_DA.jpg');
Name_DAfig=fullfile(save_Root,'IccScatterAvDiam_DA.fig');
fig=gcf;
fig.PaperPositionMode = 'auto';
print('-dpng','-r0', Name_DA);
print('-dpng','-r0', Name_DAfig);

figure('Position',[60 60 1600 820]);n=0;
for ii=8:13
    n=n+1;
    subplot(2,3,n); 
    scatter(ManualDiameters(:,n),AutomaticDiameters(:,n),'filled')
    AAposition=11-ii;
    ylabel('Av Automatic Diameter(mm)','FontSize',13)
    xlabel('Manual Diameter(mm)','FontSize',13)
    title({[num2str(AAposition),'cm AA, ICC = ', num2str(IccPerslice_AvDiam(1,ii),'%0.2f')]},'FontSize',13)
    lsline ; %Add least-squares line to scatter plot
    set(gca, 'FontSize',13,'linew',1.1) % to set axes width and increase font size
    axis equal   
end
Name_AA=fullfile(save_Root,'IccScatterAvDaim_AA.jpg');
Name_AAfig=fullfile(save_Root,'IccScatterAvDaim_AA.fig');
fig=gcf;
fig.PaperPositionMode = 'auto';        
print('-dpng','-r0', Name_AA);
print('-dpng','-r0', Name_AAfig);

%% Table of Icc per slice for average diameter
DiamAtSlice={'DA3cmbelow','DA2cmbelow','DA1cmbelow','DABifLevel','DA1cmAbove',...
    'DA2cmAbove','DA3cmAbove','AA3cmAbove','AA2cmAbove',...
    'AA1cmAbove','AABifLevel','AA1cmbelow','AA2cmbelow'};
DiamRightOrder=fliplr(DiamAtSlice);
ICCPerason_Av_RightOrder=fliplr(ICCPerasonPerslice_AvDaiam);
rowsname={'ICC','LowerBand','UpperBand','PValue','r Pearson','r2 Pearson'};
TableICC_AvDiam=array2table(ICCPerason_Av_RightOrder, 'VariableNames',DiamRightOrder,'RowNames',rowsname)
Tablename=fullfile(save_Root,'TableIccPerasonPerslice_AvDiam.txt');
writetable(TableICC_AvDiam, Tablename, 'WriteVariablenames', true,'writerownames',true);

save(fullfile(save_Root,'ICCPerasonPerslice_AvDaiam.mat'),'ICCPerason_Av_RightOrder')

%% get ICC for all slices of MAx Diamwter
% slices are -3cm DA:3cmDA,3cm AA: -2cm AA
ManualDiameters=DiamMapMan_Max(:,2:end);
AutomaticDiameters=DiamMapAuto_Max(:,2:end);

 for slice= 1:slicesNum
     diameters=[ManualDiameters(:,slice),AutomaticDiameters(:,slice)];
     [Icc, LB, UB, F, df1, df2, pvalue] = ICC_v2(diameters, 'A-1' ); 
     IccPerslice_Max(:,slice) = [Icc; LB; UB; pvalue];
     r = corrcoef(ManualDiameters(:,slice),AutomaticDiameters(:,slice));
     rPearson_Max(1,slice)=r(1,2);
     r2Pearson_Max(1,slice)=rPearson_Max(1,slice)^2;
 end
ICCPerasonPerslice_MaxDaiam=[IccPerslice_Max;rPearson_Max;r2Pearson_Max];
save(fullfile(save_Root,'IccPerslice_MaxDaim.mat'),'IccPerslice_Max')
% save(fullfile(save_Root,'ICCPerasonPerslice_MaxDaiam.mat'),'ICCPerasonPerslice_MaxDaiam')

DiametersAll=[reshape(ManualDiameters, size(ManualDiameters,1)*size(ManualDiameters,2), 1),...
    reshape(AutomaticDiameters,size(AutomaticDiameters,1)*size(AutomaticDiameters,2), 1)];
[Icc, LB, UB, F, df1, df2, pvalue] = ICC_v2(DiametersAll, 'A-1' ); 
TotalICC_Max=Icc;
save(fullfile(save_Root,'TotalICC_MaxDaim.mat'),'TotalICC_Max')
figure; scatter(DiametersAll(:,1),DiametersAll(:,2),'filled')
ylabel('Max Automatic Diameter(mm)','FontSize',13)
xlabel('Max Manual Diameter(mm)','FontSize',13)
title({[' Total ICC of amx Diameter for All Measures= ', num2str(TotalICC_Max,'%0.2f')]},'FontSize',13)
lsline ; %Add least-squares line to scatter plot
set(gca, 'FontSize',13,'linew',1.1) % to set axes width and increase font size
axis equal
Name_Tfig=fullfile(save_Root,'IccTotal_MaxDiam.jpg');
fig=gcf;
fig.PaperPositionMode = 'auto';
print('-dpng','-r0', Name_Tfig);    
%% Table of Icc per slice with max diameter
DiamAtSlice={'DA3cmbelow','DA2cmbelow','DA1cmbelow','DABifLevel','DA1cmAbove',...
    'DA2cmAbove','DA3cmAbove','AA3cmAbove','AA2cmAbove',...
    'AA1cmAbove','AABifLevel','AA1cmbelow','AA2cmbelow'};
DiamRightOrder=fliplr(DiamAtSlice);
ICCPerason_Max_RightOrder=fliplr(ICCPerasonPerslice_MaxDaiam);
rowsname={'ICC','LowerBand','UpperBand','PValue','r Pearson','r2 Pearson'}; 
TableICC_Max=array2table(ICCPerason_Max_RightOrder, 'VariableNames',DiamRightOrder,'RowNames',rowsname)
Tablename=fullfile(save_Root,'TableIccPearsonPerslice_MaxDiam.txt');
writetable(TableICC_Max, Tablename, 'WriteVariablenames', true,'writerownames',true);

save(fullfile(save_Root,'ICCPerasonPerslice_MaxDaiam.mat'),'ICCPerason_Max_RightOrder')

%%
%%%%%%%%%%%%%%%%%%% plot the scatter for each slice for Maximum Diameter %%%%%%%%%%%%%%%%%%%
ManualDiameters=DiamMapMan_Max(:,2:end);
AutomaticDiameters=DiamMapAuto_Max(:,2:end);
if true
    figure('Position',[60 60 1600 820]);
    nn=1;
    for ii=1:7
        subplot(2,4,ii);   
        scatter(ManualDiameters(:,ii),AutomaticDiameters(:,ii),'filled')
        DAposition=-4+ii;
        ylabel('Max Automatic Diameter(mm)','FontSize',13)
        xlabel('Manual Diameter(mm)','FontSize',13)
    %   title({[num2str(DAposition),' cm+LBif at DA & ICC = ', num2str(IccPerslice(1,ii),'%0.2f')]},'FontSize',13)
        title({[num2str(DAposition),' cm DA, ICC = ', num2str(IccPerslice_Max(1,ii),'%0.2f')]},'FontSize',13)
        lsline ; %Add least-squares line to scatter plot
        set(gca, 'FontSize',13,'linew',1.1) % to set axes width and increase font size
        axis equal

    end
    Name_DA=fullfile(save_Root,'IccScatterMaxDaim_DA.jpg');
    Name_DAfig=fullfile(save_Root,'IccScatterMaxDiam_DA.fig');
    fig=gcf;
    fig.PaperPositionMode = 'auto';
    print('-dpng','-r0', Name_DA);
    print('-dpng','-r0', Name_DAfig);


    for ii=1:7
    %BlandAltman Plot
        DAposition=-4+ii;
    %   tit = {['DA', num2str(DAposition),'cm from Bif Level ']}; % figure title
        tit ='';
        label = {'Manual(mm)','Automatic(mm)'}; % Names of data sets
    %   [cr, fig, statsStruct] =BlandAltman(ManualDiameters(:,ii),AutomaticDiameters(:,ii),label,tit); % becouase the data in not destrobuted normaly(gaussian) I use non- parametric distribution 'baStatsMode','Non-parametric'
        level=strjoin({num2str(DAposition) ,'cm DA' },'');
        [cr, fig, statsStruct] =BlandAltman(ManualDiameters(:,ii),AutomaticDiameters(:,ii),label,tit,{'Data','Bias'},'BAINFO',{'RPC(%)'; 'ICC';'r';'r2'},'LEVEL',level);
        set(gca, 'FontSize',13,'linew',1.1)
        nameDA={['BlanAltman_MaxDaim_',num2str(DAposition),'cmDA.jpg']};
        Name_DABland=fullfile(save_Root,nameDA{1,1});
    %   saveas(gcf,Name_DABland)
        fig=gcf;
        fig.PaperPositionMode = 'auto';
        print('-dpng','-r0', Name_DABland);
        close gcf
        PearsonICC(1,ii)=DAposition;
        PearsonICC(2,ii)=statsStruct.ICC;
        PearsonICC(3,ii)=statsStruct.r;
        PearsonICC(4,ii)=statsStruct.r2;

    end

    figure('Position',[60 60 1600 820]);n=0;
    for ii=8:13
        n=n+1;
        subplot(2,3,n); 
        scatter(ManualDiameters(:,n),AutomaticDiameters(:,n),'filled')
        AAposition=11-ii;
        ylabel('Max Automatic Diameter(mm)','FontSize',13)
        xlabel('Manual Diameter(mm)','FontSize',13)
        title({[num2str(AAposition),'cm AA, ICC = ', num2str(IccPerslice_Max(1,ii),'%0.2f')]},'FontSize',13)
    %   title({[num2str(AAposition),'cm+LPAB , ICC = ', num2str(IccPerslice(1,ii),'%0.2f')]},'FontSize',13)   
        lsline ; %Add least-squares line to scatter plot
        set(gca, 'FontSize',13,'linew',1.1) % to set axes width and increase font size
        axis equal   
    end
    Name_AA=fullfile(save_Root,'IccScatterMaxDaim_AA.jpg');
    Name_AAfig=fullfile(save_Root,'IccScatterMaxDaim_AA.fig');
    fig=gcf;
    fig.PaperPositionMode = 'auto';
    print('-dpng','-r0', Name_AA);
    print('-dpng','-r0', Name_AAfig);
    n=0;
    for ii=8:13
    %BlandAltman Plot
        AAposition=11-ii;
    %   tit = {['AA', num2str(AAposition),'cm from Bif Level ']}; % figure title
        tit ='';
        label = {'Manual (mm)','Automatic(mm)'}; % Names of data sets
    %   [cr, fig, statsStruct] =BlandAltman(ManualDiameters(:,ii),AutomaticDiameters(:,ii),label,tit);% becouase the data in not destrobuted normaly(gaussian) I use non- parametric distribution 'baStatsMode','Non-parametric'
        level=strjoin({num2str(AAposition) ,'cm AA' },'');
        [cr, fig, statsStruct] =BlandAltman(ManualDiameters(:,ii),AutomaticDiameters(:,ii),label,tit,{'Data','Bias'},'BAINFO',{'RPC(%)'; 'ICC';'r';'r2'},'LEVEL',level);
        set(gca, 'FontSize',13,'linew',1.1)
        nameAA={['BlanAltman_MaxDaim_',num2str(AAposition),'cmAA.jpg']};
        Name_AABland=fullfile(save_Root,nameAA);
    %   saveas(gcf,Name_AABland{1,1})
        fig=gcf;
        fig.PaperPositionMode = 'auto';
        print('-dpng','-r0', Name_AABland{1,1});
        close gcf
        PearsonICC(1,ii)=AAposition;
        PearsonICC(2,ii)=statsStruct.ICC;
        PearsonICC(3,ii)=statsStruct.r;
        PearsonICC(4,ii)=statsStruct.r2;
    end

end

%% %%%%%%%% table for Icc, Perason & pearson squared %%%%%%%%%%
DiamAtSlice={'DA3cmbelow','DA2cmbelow','DA1cmbelow','DABifLevel','DA1cmAbove',...
    'DA2cmAbove','DA3cmAbove','AA3cmAbove','AA2cmAbove',...
    'AA1cmAbove','AABifLevel','AA1cmbelow','AA2cmbelow'};
rowsname={'ICC','r-Pearson','r2- Pearson'};
TableICCPearson_Max=array2table(PearsonICC(2:end,:), 'VariableNames',DiamAtSlice,'RowNames',rowsname)
Tablename=fullfile(save_Root,'TablePearson_Icc_Perslice_MaxDiam.txt');
writetable(TableICCPearson_Max, Tablename, 'WriteVariablenames', true,'writerownames',true);


%% %%%%%%%%%%%%%%%%%%%scatter plot for Maximum Surface Distance and Dice %%%%%%%%%%%%%%%%%%%
f = figure('Position',[60 60 1400 820]); 
columns = {'DSC','MaxSD','MeanSD','stdSD','Jaccard'};
rows ={'Mean','STD','Min','Max'};
Variable_list_nonNan=Variable_list;
Variable_list_nonNan(isnan(Variable_list_nonNan))=0;
Variable_list_nonNan( all(~Variable_list_nonNan(:,3:end),2),:) = [];%
data(1,:)=mean(Variable_list_nonNan(:,2:end));
data(2,:)=std(Variable_list_nonNan(:,2:end));
data(3,:)=min(Variable_list_nonNan(:,2:end));
data(4,:)=max(Variable_list_nonNan(:,2:end));
tabl = uitable('Data',data,'ColumnName',columns,'RowName',rows,...
'Parent',f,'Position',[981 410 419 102],'FontSize',10);
DiceStatistics=data;
save(fullfile(save_Root,'DiceStatistics.mat'),'DiceStatistics');

Table_DiceStatistics=table(DiceStatistics(1,:)',DiceStatistics(2,:)',...
    DiceStatistics(3,:)',DiceStatistics(4,:)','RowNames',columns,'VariableNames',rows)
writetable(Table_DiceStatistics,fullfile(save_Root,'Table_DiceStatistics.txt'),'WriteRowNames',true);

VolID=Variable_list(:,1);
volumsNum=size(Variable_list,1);
subplot(3, 1, 1); % comp Dice
hold on;s1=scatter(1:volumsNum,Variable_list(:,2),20,'b','filled');
s1.MarkerEdgeColor = [0 .5 .5];s1.SizeData=45; s1.LineWidth=2;s1.MarkerFaceColor=[0 .8 .8];legend('-DynamicLegend');
hold on;plot(1:volumsNum,ones(1,volumsNum).*mean(Variable_list(:,2)),'--r','MarkerSize',15);
title('Dice of Complete volume of 100 CT(per optimal sets)');
legend('-DynamicLegend','Dice OptSet',...
    ['mean DSC=',num2str(mean(Variable_list(:,2)),'%.3f')],'Location','Best');
ylabel('Dice');set(gca, 'FontSize',13,'linew',1.2);
ytic=[round(min(Variable_list(:,2)),2,'significant'):.02:max(Variable_list(:,2))+0.01]';
set(gca,'YTick',ytic,'yTickLabel',num2str(ytic));
xlim([0,volumsNum+1])
Xtickle=[1:volumsNum];
set(gca,'XTick',Xtickle,'XTickLabel',num2str(VolID));
grid on; set(gca, 'position', [0.05 0.72 0.65 0.24] );

hold on;subplot(3, 1,2); % comp MAx SD
hold on;s2=scatter(1:volumsNum,Variable_list(:,4),20,'b','filled');
s2.MarkerEdgeColor = [0.1 .5 .5];s2.SizeData=45; s2.LineWidth=2;s2.MarkerFaceColor=[0.8 .2 .8];legend('-DynamicLegend');
hold on;plot(1:volumsNum,ones(1,volumsNum).*data(1,3),'--r','LineWidth',.5,'MarkerSize',6,'MarkerEdgeColor','r','MarkerFaceColor',[.8 .2 .2])
legend('-DynamicLegend','Mean SD OptSet',['mean MeanSD=',num2str(data(1,3),'%.3f')],'Location','Best')
title('Mean Surface distance)');
ylabel('Mean SD')
ylim([0,0.3+data(4,3)])
xlim([0,volumsNum+1])
Xtickle=[1:volumsNum];
set(gca,'XTick',Xtickle,'XTickLabel',num2str(VolID));
grid on;set(gca, 'position', [0.05 0.39 0.65 0.24] );set(gca, 'FontSize',13,'linew',1.2);

hold on;subplot(3, 1,3); % Comp Mean SD
hold on;s3=scatter(1:volumsNum,Variable_list(:,3),20,'r','filled');
s3.MarkerEdgeColor=[0.2 .3 .1];s3.SizeData=45; s3.LineWidth=2; s3.MarkerFaceColor=[0.7 0.3 0.2];legend('-DynamicLegend');
hold on;plot(1:volumsNum,ones(1,volumsNum).*data(1,2),'--r','LineWidth',.5,'MarkerSize',6,'MarkerEdgeColor','r','MarkerFaceColor',[.8 .2 .2])
legend('-DynamicLegend','Max SD OptSet',['mean MaxSD=',num2str(data(1,2),'%.3f')],'Location','Best')
title('Max Surface distance)');
ylabel('Max SD')
ylim([0,0.3+data(4,2)])
xlim([0,volumsNum+1])
Xtickle=[1:volumsNum];
set(gca,'XTick',Xtickle,'XTickLabel',num2str(VolID));
grid on;set(gca, 'position', [0.05 0.06 0.65 0.24] );set(gca, 'FontSize',13,'linew',1.2);
fig=gcf;
fig.PaperPositionMode = 'auto';
print('-dpng','-r0', fullfile(save_Root,'DiceMSDOpt3sets.jpg'));

%% %%%%%%%%%%%%%%%%%%% Outliers and Box plot for mean & Maximum Surface Distance and Dice %%%%%%%%%%%%%%%%%%%
figure('Position',[50 50 1300 350])
DCS=Variable_list(:,2);
subplot(1,3,1); bh1=boxplot(DCS,'Labels',{'DSC'},'Whisker',1,'Width',0.2);
% extract outliers
Outliers_Max= findobj(bh1,'Tag','Outliers');  %to get the outliers
yy_Max = get(Outliers_Max,'YData');
uW_Max=get(findobj(bh1,'Tag','Upper Whisker'),'YData');
LW_Max=get(findobj(bh1,'Tag','Lower Whisker'),'YData');
Autliers_BoxPlot_MaxDiam_Dice=[];
[logics,locs]=find(ismember(DCS',yy_Max));
vols=Variable_list(locs,1);
Autliers_BoxPlot_MaxDiam_Dice=[vols,DCS(locs)];
[Autliers_Unique_MaxDiam_Dice, label , distribut]= unique(Autliers_BoxPlot_MaxDiam_Dice); 
save(fullfile(save_Root,'Autliers_Unique_MaxDiam_Dice.mat'),'Autliers_Unique_MaxDiam_Dice');
save(fullfile(save_Root,'Autliers_BoxPlot_MaxDiam_Dice.mat'),'Autliers_BoxPlot_MaxDiam_Dice');
title( 'Dice of Aorta','FontSize',18)
ylabel ('Dice','FontSize', 18)
text(1.15,round(mean(DCS),2),num2str(round(mean(DCS),2)))
set(gca, 'FontSize',18,'linew',1.7)
txt = findobj(gca,'Type','text');
set(txt(1:end),'VerticalAlignment', 'Middle','FontSize',18);
set(bh1,'linewidth',1.7);

MSD=Variable_list(:,3:4);
MSD(isnan(MSD))=0;
MSD( all(~MSD,2),:) = [];%
subplot(1,3,2);bh2=boxplot(MSD(:,1),'Labels',{'MaxSD'},'Whisker',1,'Width',0.2);
title( 'Max Surface Distance','FontSize',18)
ylabel ('MaxSD (mm)','FontSize', 18)
text(1.15,round(mean(MSD(:,1)),2),num2str(round(mean(MSD(:,1)),2)))
set(gca, 'FontSize',18,'linew',1.7)
txt = findobj(gca,'Type','text');
set(txt(1:end),'VerticalAlignment', 'Middle','FontSize',18);
set(bh2,'linewidth',1.7);

subplot(1,3,3);bh3=boxplot(MSD(:,2),'Labels',{'MeanSD'},'Whisker',1,'Width',0.2);
title( 'Mean Surface Distance','FontSize',18)
ylabel ('MeanSD (mm)','FontSize', 18)
text(1.15,round(mean(MSD(:,2)),2),num2str(round(mean(MSD(:,2)),2)))
set(gca, 'FontSize',18,'linew',1.7)
txt = findobj(gca,'Type','text');
set(txt(1:end),'VerticalAlignment', 'Middle','FontSize',18);
set(bh3,'linewidth',1.7);

NameMSD=fullfile(save_Root,'DSC_MSD_boxplot_Aorta.jpg');
Namefig=fullfile(save_Root,'DSC_MSD_boxplot_Aorta.fig');
fig=gcf;
fig.PaperPositionMode = 'auto';
print('-dpng','-r0', NameMSD);
print('-dpng','-r0', Namefig);

%% Diameters per slice
f = figure('Position',[10 40 1525 820]); 
columns ={'DiamDifff','AutoMean','ManMean','AutoSTD','ManSTD'};  %'AutoMax','ManMax','AutoMin','ManMin'};
rows ={'-3DA','-2DA','-1DA','DABif','+1DA','+2DA','+3DA','+3AA',...
    '+2AA','+1AA','AABif','-1AA','-2AA'};
ManDiamStat=[mean(DiamMapMan_Max(:,2:end))',std(DiamMapMan_Max(:,2:end))',max(DiamMapMan_Max(:,2:end))',min(DiamMapMan_Max(:,2:end))'];
AutoDiamStat=[mean(DiamMapAuto_Max(:,2:end))',std(DiamMapAuto_Max(:,2:end))',max(DiamMapAuto_Max(:,2:end))',min(DiamMapAuto_Max(:,2:end))'];
diffdiam=ManDiamStat(:,1)-AutoDiamStat(:,1); %- means Automatic is larger
DiamStats=[diffdiam,AutoDiamStat(:,1),ManDiamStat(:,1),AutoDiamStat(:,2),ManDiamStat(:,2),...
    AutoDiamStat(:,3),ManDiamStat(:,3),AutoDiamStat(:,4),ManDiamStat(:,4)];
save(fullfile(save_Root,'DiamStats.mat'),'DiamStats')
data2=DiamStats(:,1:5); 
tabl = uitable('Data',data2,'ColumnName',columns,'RowName',rows,...
'Parent',f,'Position',[1140 180 370 298],'FontSize',9);
uicontrol('Style', 'text', 'Position', [1110 480 340 20], 'String', 'Average diameter per slice in Manual and automatic segmentations');

TableDiamStat=array2table(data2, 'VariableNames',columns,'RowNames',rows)
Tablename=fullfile(Root_Folder,'TableDiamStat.txt');
writetable(TableDiamStat, Tablename, 'WriteVariablenames', true,'writerownames',true);

% diameter error per volume
ErrorAllSlices=abs(ErrorMaxDiam);
ErrorAllSlices( all(~ErrorAllSlices(:,2:end),2),:) = [];
ErrorAvreagevolume(:,1)=ErrorAllSlices(:,1);
ErrorAvreagevolume(:,2:5)=[mean(ErrorAllSlices(:,2:end),2),max(ErrorAllSlices(:,2:end),[],2),min(ErrorAllSlices(:,2:end),[],2),std(ErrorAllSlices(:,2:end),[],2)];
save(fullfile(save_Root,'ErrorMaxDiamAvragevolume.mat'),'ErrorAvreagevolume')

volumsNum=size(ErrorAvreagevolume,1);
subplot(3, 1, 1); % diam error for all volums
hold on;s1=scatter(1:volumsNum,ErrorAvreagevolume(:,2),20,'b','filled');
s1.MarkerEdgeColor = [0 .5 .5];s1.SizeData=45; s1.LineWidth=2;s1.MarkerFaceColor=[0 .8 .8];legend('-DynamicLegend');
hold on;plot(1:volumsNum,ones(1,volumsNum).*mean(ErrorAvreagevolume(:,2)),'--r','LineWidth',.5,'MarkerSize',6,'MarkerEdgeColor','r','MarkerFaceColor',[.8 .2 .2])
title('Average Max Diameter error for each volum over 15 slices on 100 CT');
legend('-DynamicLegend',['Av DiamError, mean=',num2str(mean(ErrorAvreagevolume(:,2)),'%.3f')],'Location','Best');
ylabel('Diam Error(mm)');
ytic=[0:.5:max(ErrorAvreagevolume(:,2))+1]';
set(gca,'YTick',ytic,'yTickLabel',num2str(ytic));
set(gca, 'FontSize',13,'linew',1.2);
xlim([0,volumsNum+1])
Xtickle=[1:volumsNum]';
set(gca,'XTick',Xtickle,'XTickLabel',num2str(VolID));
grid on; set(gca, 'position', [0.05 0.72 0.65 0.24] );

volumsNum=size(ErrorAvreagevolume,1);
subplot(3, 1, 2); % comp Dice
hold on;s2=scatter(1:volumsNum,ErrorAvreagevolume(:,3),20,'r','filled');
s2.MarkerEdgeColor = [0.8 .5 .5];s2.SizeData=45; s2.LineWidth=2;s2.MarkerFaceColor=[0.8 0.2 0.2];legend('-DynamicLegend');
hold on;plot(1:volumsNum,ones(1,volumsNum).*mean(ErrorAvreagevolume(:,3)),'--r','LineWidth',.5,'MarkerSize',6,'MarkerEdgeColor','r','MarkerFaceColor',[.8 .2 .2])
title('Maximum Diameter error for each volum over 15 slices on 100 CT');
legend('-DynamicLegend',['Max DiamError, mean=',num2str(mean(ErrorAvreagevolume(:,3)),'%.3f')],'Location','Best');
ylabel('Diam Error(mm)');ylim([0,max(ErrorAvreagevolume(:,3))+1])
ytic=[1:2:max(ErrorAvreagevolume(:,3))+1]';
set(gca,'YTick',ytic,'yTickLabel',num2str(ytic));
xlim([0,volumsNum+1])
Xtickle=[1:volumsNum]';
set(gca,'XTick',Xtickle,'XTickLabel',num2str(VolID));
set(gca, 'FontSize',13,'linew',1.2);
grid on;set(gca, 'position', [0.05 0.39 0.65 0.24] );set(gca, 'FontSize',13,'linew',1.2);

% Error per slice over all 100 ct
ErrorAvreageslices=[mean(ErrorAllSlices(:,2:end),1);max(ErrorAllSlices(:,2:end),[],1);min(ErrorAllSlices(:,2:end),[],1);std(ErrorAllSlices(:,2:end),[],1)];
save(fullfile(save_Root,'ErrorAvreageslices.mat'),'ErrorAvreageslices')
colnames={'Mean','Max','Min','std'};
rowsname={'Mean AvAllVolover15slices','Max AvAllVolover15slices','-3DAAv','-2DAAv','-1DAAv','DABifAv','+1DAAv',...
    '+2DAAv','+3DAAv','+3AAAv','+2AAAv','+1AAAv','AABifAv','-1AAAv','-2AAAv'};
errorStats=[mean(ErrorAvreagevolume(:,2:3))',max(ErrorAvreagevolume(:,2:3))',min(ErrorAvreagevolume(:,2:3))',std(ErrorAvreagevolume(:,2:3))'];
errorStats=[errorStats;ErrorAvreageslices'];
Tableerror=array2table(errorStats, 'VariableNames',colnames,'RowNames',rowsname)
Tablename=fullfile(save_Root,'TableErrorStatistics.txt');
writetable(Tableerror, Tablename, 'WriteVariablenames', true,'writerownames',true);
save(fullfile(save_Root,'ErrorStats.mat'),'errorStats')

volumsNum=size(ErrorAvreageslices,2);
subplot(3, 1, 3); % Max and Mean Diameter Per slice
hold on;s3=scatter(1:volumsNum,ErrorAvreageslices(1,:),20,'r','filled');
s3.MarkerEdgeColor = [0 .5 .5];s3.SizeData=45; s3.LineWidth=2;s3.MarkerFaceColor=[0 0.8 0.8];legend('-DynamicLegend');
hold on;s4=scatter(1:volumsNum,ErrorAvreageslices(2,:),20,'r','filled');
s4.MarkerEdgeColor = [0.8 .5 .5];s4.SizeData=45; s4.LineWidth=2;s4.MarkerFaceColor=[0.8 0.2 0.2];legend('-DynamicLegend');
title('Average Diameter error per slice over 100 volume');
legend('-DynamicLegend',['Av DiamError, mean=',num2str(mean(ErrorAvreageslices(1,:)),'%.3f')],...
    ['Max DiamError, mean=',num2str(mean(ErrorAvreageslices(2,:)),'%.3f')],'Location','Best');
ylabel('Diam Error(mm)');ylim([0,max(ErrorAvreageslices(2,:))+1])
set(gca, 'FontSize',13,'linew',1.2);
xlim([0,volumsNum+1])
Xtickle=[1:volumsNum]';
set(gca,'XTick',Xtickle,'XTickLabel',num2str(VolID));
grid on;set(gca, 'position', [0.05 0.04 0.65 0.27] );set(gca, 'FontSize',13,'linew',1.2);
grid minor
xtic={'-3DA','-2DA','-1DA','DABif','+1DA','+2DA','+3DA','+3AA',...
    '+2AA','+1AA','AABif','-1AA','-2AA'};
set(gca,'XTick',1:length(xtic),'XTickLabel',xtic);
ytic=[1:2:max(ErrorAvreageslices(2,:))+1]';
set(gca,'YTick',ytic,'yTickLabel',num2str(ytic));
fig=gcf;
fig.PaperPositionMode = 'auto';
print('-dpng','-r0', fullfile(save_Root,'DiamErrorOpt.jpg'));

%%  %%%%%%%%%%%%%%%%%%% Box plot Max Diameter Error per slice %%%%%%%%%%%%%%%%%%%
 figure('Position',[70 70 1800 650])
 DiamErrMax=abs(ErrorMaxDiam(:,2:end));
 DiamErrMax=fliplr(DiamErrMax);
  xtic={'-2AA','-1AA','AABif','+1AA','+2AA','+3AA','+3DA','+2DA','+1DA',...
  'DABif','-1DA','-2DA','-3DA'};

bherr=boxplot(DiamErrMax,'Labels',xtic,'Whisker',1,'Width',0.7);
Outliers_DiamMax= findobj(bherr,'Tag','Outliers');  %to get the outliers
yy_Max = get(Outliers_DiamMax,'YData');
uW_Max=get(findobj(bherr,'Tag','Upper Whisker'),'YData');
LW_Max=get(findobj(bherr,'Tag','Lower Whisker'),'YData');
All_Autliers_MaxDiam=[];
for ii=1:slicesNum
    [logics,locs]=find(ismember(DiamErrMax(:,ii)',yy_Max{ii,1}));
    vols=ErrorMaxDiam(locs,1);
    voloutlier=[vols,DiamErrMax(locs,ii)]
    All_Autliers_MaxDiam=[All_Autliers_MaxDiam;voloutlier];
end
[All_Autliers_Unique_MaxDiam, label , distribut]= unique(All_Autliers_MaxDiam(:,1)); 
save(fullfile(save_Root,'All_Autliers_Unique_MaxDiam.mat'),'All_Autliers_Unique_MaxDiam');
save(fullfile(save_Root,'All_Autliers_MaxDiam.mat'),'All_Autliers_MaxDiam');

title( 'Maximum Diameter Difference Per Measuring Level','FontSize',18)
ylabel ('Error (mm)','FontSize', 18)
set(gca, 'FontSize',18,'linew',1.7)
txt = findobj(gca,'Type','text');
set(txt(1:end),'VerticalAlignment', 'Middle','FontSize',18);
for w=1:size(bherr,2)
     set(bherr(:,w),'linewidth',1.7);
     text(w-0.1,round(mean(DiamErrMax(:,w)),2)-0.08,num2str(round(median(DiamErrMax(:,w)),2)),'FontSize',14)
end
set(gca, 'position', [0.04 0.08 0.94 0.86] );
NameErr=fullfile(save_Root,'DiamError_MaxDiam_boxplot.jpg');
NameErrfig=fullfile(save_Root,'DiamError_MaxDiam_boxplot.fig');
fig=gcf;
fig.PaperPositionMode = 'auto'; 
print('-dpng','-r0', NameErr);
print('-dpng','-r0', NameErrfig);


%%  %%%%%%%%%%%%%%%%%%% Box plot Average Diameter Error per slice %%%%%%%%%%%%%%%%%%%
 figure('Position',[70 70 1800 650])
 DiamErr=abs(ErrorAvDiam(:,2:end));
 DiamErr=fliplr(DiamErr);
  xtic={'-2AA','-1AA','AABif','+1AA','+2AA','+3AA','+3DA','+2DA','+1DA',...
  'DABif','-1DA','-2DA','-3DA'};

bherr=boxplot(DiamErr,'Labels',xtic,'Whisker',1,'Width',0.7);
Outliers_DiamMax= findobj(bherr,'Tag','Outliers');  %to get the outliers
yy_Max = get(Outliers_DiamMax,'YData');
uW_Max=get(findobj(bherr,'Tag','Upper Whisker'),'YData');
LW_Max=get(findobj(bherr,'Tag','Lower Whisker'),'YData');
All_Autliers_AvDiam=[];
for ii=1:slicesNum
    [logics,locs]=find(ismember(DiamErr(:,ii)',yy_Max{ii,1}));
    vols=ErrorAvDiam(locs,1);
    voloutlier=[vols,DiamErr(locs,ii)];
    All_Autliers_AvDiam=[All_Autliers_AvDiam;voloutlier];
end
[All_Autliers_Unique_AvDiam, label , distribut]= unique(All_Autliers_AvDiam(:,1)); 
save(fullfile(save_Root,'All_Autliers_Unique_AvDiam.mat'),'All_Autliers_Unique_AvDiam');
save(fullfile(save_Root,'All_Autliers_AvDiam.mat'),'All_Autliers_AvDiam');

title( 'Average Diameter Difference Per Measuring Level','FontSize',18)
ylabel ('Error (mm)','FontSize', 18)
set(gca, 'FontSize',18,'linew',1.7)
txt = findobj(gca,'Type','text');
set(txt(1:end),'VerticalAlignment', 'Middle','FontSize',18);
for w=1:size(bherr,2)
     set(bherr(:,w),'linewidth',1.7);
     text(w-0.1,round(mean(DiamErr(:,w)),2)-0.08,num2str(round(median(DiamErr(:,w)),2)),'FontSize',14)
end
set(gca, 'position', [0.04 0.08 0.94 0.86] );
NameErr=fullfile(save_Root,'DiamError_AvDiam_boxplot.jpg');
NameErrfig=fullfile(save_Root,'DiamError_AvDiam_boxplot.fig');
fig=gcf;
fig.PaperPositionMode = 'auto'; 
print('-dpng','-r0', NameErr);
print('-dpng','-r0', NameErrfig);

%% errors larger than 4mm
[x,y]=find(ErrorMaxDiam(:,2:end)>4)
 vol_LargerErrors_Max=unique(ErrorMaxDiam(x,1));
 errorpercentage=numel(vol_LargerErrors_Max)/size(ErrorMaxDiam,1)
 save(fullfile(save_Root,'volumes_LargerErrors_Max.mat'),'vol_LargerErrors_Max')
 save(fullfile(save_Root,'Errorpercentage_MaxDiam_Volumes.mat'),'errorpercentage')

[x2,y2]=find(ErrorAvDiam(:,2:end)>4)
 vol_LargerErrors_Av=unique(ErrorAvDiam(x,1));
%% find the maximum diameter errors
DiamErr=abs(ErrorMaxDiam(:,2:end));
[maxval,maxloc]=max(DiamErr);
AvDiamError=mean(DiamErr(:));
STDDiamError=std(DiamErr(:));
MaxDiamError=max(DiamErr(:));
MinDiamError=min(DiamErr(:));
MaxDiamerror_PerSlice(1,1:13)=[-3:3,3:-1:-2];
MaxDiamerror_PerSlice(2,1:13)=maxval;
MaxDiamerror_PerSlice(3,1:13)=ErrorAvDiam(maxloc,1);
save(fullfile(save_Root,'Volumes_MaxDiamerror_PerSlice.mat'),'MaxDiamerror_PerSlice')
DiamErrorAllMeasure_Stats=[AvDiamError,STDDiamError,MaxDiamError,MinDiamError];
save(fullfile(save_Root,'DiamErrorAllMeasure_Stats.mat'),'DiamErrorAllMeasure_Stats')

%% plot Average Diameter of manual and automatic measures per slice
% f = figure('Position',[10 40 1525 820]);
% SlicNum=size(ManualDiameters,2);
% s1=scatter(1:SlicNum,mean(ManualDiameters),20,'b','filled');
% s1.MarkerEdgeColor = [0 .5 .5];s1.SizeData=45; s1.LineWidth=2;s1.MarkerFaceColor=[0 .8 .8];legend('-DynamicLegend');
% lower=mean(ManualDiameters)-min(ManualDiameters);
% upper=max(ManualDiameters)-mean(ManualDiameters);
% hold on;errorbar(1:SlicNum,mean(ManualDiameters),lower,upper,'.b','LineWidth',1.2,'MarkerSize',6,'MarkerEdgeColor','r','MarkerFaceColor',[.8 .2 .2])
% hold on;s4=scatter(1:SlicNum,mean(AutomaticDiameters),20,'r','filled');
% s4.MarkerEdgeColor = [0.8 .5 .5];s4.SizeData=45; s4.LineWidth=2;s4.MarkerFaceColor=[0.8 0.2 0.2];legend('-DynamicLegend');
% lowerA=mean(AutomaticDiameters)-min(AutomaticDiameters);
% upperA=max(AutomaticDiameters)-mean(AutomaticDiameters);
% hold on;errorbar(1:SlicNum,mean(AutomaticDiameters),lowerA,upperA,'.r','LineWidth',1.2,'MarkerSize',6,'MarkerEdgeColor','r','MarkerFaceColor',[.8 .2 .2])
% title('Mean Manual& Automatic Diameter Per Slice of 100 CT');
% legend('-DynamicLegend','Manual','Manual Errorbar','Automatic','Automatic Errorbar','Location','Best');
% ylabel('Diameter(mm)');
% set(gca, 'FontSize',15,'linew',1.5);
% xtic={'-3DA','-2DA','-1DA','DABif','+1DA','+2DA','+3DA','+4DA','+4AA','+3AA',...
%     '+2AA','+1AA','AABif','-1AA','-2AA'};
% xlabel('Data')
% xlim([1,SlicNum])
% set(gca,'XTick',[1:SlicNum]','XTickLabel',xtic)
% set(gca, 'FontSize',15,'linew',1.5);
% grid on;
% fig=gcf;
% fig.PaperPositionMode = 'auto';
% print('-dpng','-r0', fullfile(save_Root,'Man_Auto_DiamErrorbar.jpg'));
% print('-dpng','-r0', fullfile(save_Root,'Man_Auto_DiamErrorbar.jpg'));

%% %%%%%%%%%%%%%%%%%%% Box plot of 2 paired manual and automatic diameters per slice %%%%%%%%%%%%%%%%%%%
f = figure('Position',[10 40 1525 820],'color',[1,1,1]);
aa=DiamMapAuto_Max(:,2:end);
 aa=fliplr(aa); % only 13 slices starting from -2cm AA to 3cm DA
mm=DiamMapMan_Max(:,2:end);
 mm=fliplr(mm);

Data(:,1:2:26)=aa; 
Data(:,2:2:26)=mm;
xtic={'-2AA','-1AA','AABif','+1AA','+2AA','+3AA','+3DA','+2DA','+1DA',...
  'DABif','-1DA','-2DA','-3DA'};
bherr=boxplot(Data,{reshape(repmat('A':'M',2,1),26,1) repmat((1:2)',13,1)},'factorgap',10,'color','rk') %,'Whisker',1,'Width',0.7
set(gca,'xtick',1.5:4.47:100)
set(gca,'xticklabel',xtic)

title( 'Manual and Automatic Diameters Per Measuring Level','FontSize',18)
ylabel ('Diameter (mm)','FontSize', 18)
 xlabel ('Measuring Cross-sectional Level' )
legend(findobj(gca,'Tag','Box'),'Manual','Automatic','Location','Best')
set(gca, 'FontSize',18,'linew',1.7)
txt = findobj(gca,'Type','text');
set(txt(1:end),'VerticalAlignment', 'Middle','FontSize',18);
for w=1:size(bherr,2)
     set(bherr(:,w),'linewidth',1.7);
%      text(w-0.1,round(mean(Data(:,w)),2)-0.08,num2str(round(mean(Data(:,w)),2)),'FontSize',14)
end
set(gca, 'position', [0.042 0.08 0.94 0.86] );
NameErr=fullfile(save_Root,'DiamPairs_boxplot.jpg');
NameErrfig=fullfile(save_Root,'DiamPairs_boxplot.fig');
fig=gcf;
fig.PaperPositionMode = 'auto';
print('-dpng','-r0', NameErr);
print('-dpng','-r0', NameErrfig);

