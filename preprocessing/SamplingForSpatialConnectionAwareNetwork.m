clc
clear
Filespath = '../datafile/' ;
train_axialThinpath = [Filespath, 'axialThin-train.txt'] ;
train_axialThickpath = [Filespath, 'axialThick-train.txt'] ;
train_sagittalThickspath = [Filespath, 'sagittalThicks-train.txt'] ;

axialThick = importdata(train_axialThickpath) ;
axialThin = importdata(train_axialThinpath) ;
sagittalThicks = importdata(train_sagittalThickspath) ;

BrainStructureAwareModelResults = '../output/' ;

train_case = length(axialThin) ;
datasetlength = [length(axialThin)] ;

for k = 1:train_case
    
    if rem(k-1, 30) == 0
        total_Axial2 = zeros(datasetlength(floor(k/30)+1),360,432,140) ;
        total_Reconstruction = zeros(datasetlength(floor(k/30)+1),360,432,140) ; % Memory
        total_Sag = zeros(datasetlength(floor(k/30)+1),360,432,280) ;
        total_Axial3 = zeros(datasetlength(floor(k/30)+1),360,432,140) ; % original 360 432 140
    end
    
    Img_axial2 = load(axialThick{k}) ;
    Ima = Img_axial2.T1_image ;
    Img_sagittal2 = load(sagittalThicks{k}) ;
    Im2s2 = Img_sagittal2.T2s2_image ;
    Img_3D = load(axialThin{k}) ;
    Im3 = Img_3D.T3_image ;
    Im_Brains = load([BrainStructureAwareModelResults, 'train', num2str(k), '/Reconstruction_BrainStructureAwareModel.mat']) ;
    Imr = Im_Brains.Reconstruction ;
    
    disp(['Sampling from case ', num2str(k)]) ;
    
    if size(Im2s2,3)<2*size(Imr,3)
        Im2s2 = cat(3, Im2s2, Im2s2(:,:,end)) ;
    end
    if size(Im3,3)>=140
        total_Axial2(k,:,:,:) = Ima(round(size(Im3,1)/2)-179:round(size(Im3,1)/2)+180,round(size(Im3,2)/2)-215:round(size(Im3,2)/2)+216, round(size(Im3,3)/2)-69:round(size(Im3,3)/2)+70) ;
        total_Reconstruction(k,:,:,:) = Imr(round(size(Im3,1)/2)-179:round(size(Im3,1)/2)+180,round(size(Im3,2)/2)-215:round(size(Im3,2)/2)+216, round(size(Im3,3)/2)-69:round(size(Im3,3)/2)+70) ;
        total_Axial3(k,:,:,:) = Im3(round(size(Im3,1)/2)-179:round(size(Im3,1)/2)+180,round(size(Im3,2)/2)-215:round(size(Im3,2)/2)+216, round(size(Im3,3)/2)-69:round(size(Im3,3)/2)+70) ;
        total_Sag(k,:,:,:) = Im2s2(round(size(Im3,1)/2)-179:round(size(Im3,1)/2)+180,round(size(Im3,2)/2)-215:round(size(Im3,2)/2)+216, round(size(Im2s2,3)/2)-139:round(size(Im2s2,3)/2)+140) ;
    else
        total_Axial2(k,:,:,floor((140-size(Im3,3))/2)+1:140-ceil((140-size(Im3,3))/2)) = Ima(round(size(Im3,1)/2)-179:round(size(Im3,1)/2)+180,round(size(Im3,2)/2)-215:round(size(Im3,2)/2)+216, :) ;
        total_Reconstruction(k,:,:,floor((140-size(Im3,3))/2)+1:140-ceil((140-size(Im3,3))/2)) = Imr(round(size(Im3,1)/2)-179:round(size(Im3,1)/2)+180,round(size(Im3,2)/2)-215:round(size(Im3,2)/2)+216, :) ;
        total_Axial3(k,:,:,floor((140-size(Im3,3))/2)+1:140-ceil((140-size(Im3,3))/2)) = Im3(round(size(Im3,1)/2)-179:round(size(Im3,1)/2)+180,round(size(Im3,2)/2)-215:round(size(Im3,2)/2)+216, :) ;
        total_Sag(k,:,:,floor((280-size(Im2s2,3))/2)+1:280-ceil((280-size(Im2s2,3))/2)) = Im2s2(round(size(Im3,1)/2)-179:round(size(Im3,1)/2)+180,round(size(Im3,2)/2)-215:round(size(Im3,2)/2)+216, :) ;
    end
    
    if k == length(axialThin)
        disp(['#################### Saving mat ', num2str(k), ' ####################']) ;
        
        total_Axial2 = permute(total_Axial2,[3,2,4,1]);
        T0savename = ['./SamplingForSpatialConnectionAwareNetwork/train-1-total_Axial2'] ;
        save(T0savename, 'total_Axial2', '-v7.3')
        clear total_Axial2
      
        total_Reconstruction = permute(total_Reconstruction,[3,2,4,1]);
        T1savename = ['./SamplingForSpatialConnectionAwareNetwork/train-1-total_Reconstruction'] ;
        save(T1savename, 'total_Reconstruction', '-v7.3')
        clear total_Reconstruction
        
        total_Sag = permute(total_Sag,[3,2,4,1]);
        T2savename = ['./SamplingForSpatialConnectionAwareNetwork/train-1-totalSag'] ;
        save(T2savename, 'total_Sag', '-v7.3')
        clear total_Sag

        total_Axial3 = permute(total_Axial3,[3,2,4,1]);
        T3savename = ['./SamplingForSpatialConnectionAwareNetwork/train-1-total_Axial3'] ;
        save(T3savename, 'total_Axial3', '-v7.3')
        clear total_Axial3
    end
end
