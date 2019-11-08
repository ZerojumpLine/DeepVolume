clc
clear

Filespath = '../datafile/' ;
train_axialThinpath = [Filespath, 'axialThin-train.txt'] ;
train_axialThickpath = [Filespath, 'axialThick-train.txt'] ;
train_sagittalThickpath = [Filespath, 'sagittalThick-train.txt'] ;
train_GMThinpath = [Filespath, 'GMThin-train.txt'] ;
train_WMThinpath = [Filespath, 'WMThin-train.txt'] ;

train_axialThin = importdata(train_axialThinpath) ;
train_axialThick = importdata(train_axialThickpath) ;
train_sagittalThick = importdata(train_sagittalThickpath) ;
train_GMThin = importdata(train_GMThinpath) ;
train_WMThin = importdata(train_WMThinpath) ;

datasetlength = [length(train_axialThin)] ;

samplesize = 100 ;

for k = 1:length(train_axialThin)
    
    if rem(k-1, 30) == 0
        IIin1 = zeros(samplesize*datasetlength(floor(k/datasetlength)+1), 64, 64, 64) ;
        IIin2 = zeros(samplesize*datasetlength(floor(k/datasetlength)+1), 64, 64, 64) ;
        IOout = zeros(samplesize*datasetlength(floor(k/datasetlength)+1), 64, 64, 64) ;
        IOseg = false(samplesize*datasetlength(floor(k/datasetlength)+1), 64, 64, 64, 2) ;
        IOsegsave = false(samplesize*datasetlength(floor(k/datasetlength)+1), 64, 64, 64, 3) ;
    end
        
    Img_axial = load(train_axialThick{k}) ;
    Im1 = Img_axial.T1_image ;
    Img_sagittal = load(train_sagittalThick{k}) ;
    Im2 = Img_sagittal.T2_image ;
    Img_3D = load(train_axialThin{k}) ;
    Im3 = Img_3D.T3_image ;
    Seg_1 = load_nii(train_GMThin{k}) ;
    seg1 = Seg_1.img ;
    Seg_2 = load_nii(train_WMThin{k}) ;
    seg2 = Seg_2.img ;
    
    
    [size1, size2, size3] = size(Im1) ;
    x_index = round(rand(1,samplesize)*(size1-64)+33) ;
    y_index = round(rand(1,samplesize)*(size2-64)+33) ;
    z_index = round(rand(1,samplesize)*(size3-64)+33) ;
    
    disp(['Sampling from case ', num2str(k)]) ;
    for j = 1:samplesize
        Iin1(j,:,:,:,1) = Im1(x_index(j)-32:x_index(j)+31, y_index(j)-32:y_index(j)+31, z_index(j)-32:z_index(j)+31) ;
        Iin2(j,:,:,:,1) = Im2(x_index(j)-32:x_index(j)+31, y_index(j)-32:y_index(j)+31, z_index(j)-32:z_index(j)+31) ;
        Iout(j,:,:,:,1) = Im3(x_index(j)-32:x_index(j)+31, y_index(j)-32:y_index(j)+31, z_index(j)-32:z_index(j)+31) ;
        Iseg(j,:,:,:,1) = seg1(x_index(j)-32:x_index(j)+31, y_index(j)-32:y_index(j)+31, z_index(j)-32:z_index(j)+31)>0.5 ;
        Iseg(j,:,:,:,2) = seg2(x_index(j)-32:x_index(j)+31, y_index(j)-32:y_index(j)+31, z_index(j)-32:z_index(j)+31)>0.5 ;
    end
    
    
    IIin1((k-1)*samplesize+1:k*samplesize,:,:,:) = Iin1 ;
    IIin2((k-1)*samplesize+1:k*samplesize,:,:,:) = Iin2 ;
    IOout((k-1)*samplesize+1:k*samplesize,:,:,:) = Iout ;
    IOseg((k-1)*samplesize+1:k*samplesize,:,:,:,:) = Iseg ;
    IOsegsave(:,:,:,:,1:2) = IOseg ;
    IOsegsave(:,:,:,:,3) = 1-IOseg(:,:,:,:,1)-IOseg(:,:,:,:,2) ;
    
    if k == length(train_axialThin)
        disp(['#################### Saving mat ', num2str(k), ' ####################']) ;
        list = randperm(size(IOout,1)) ;
        T1r = IIin1(list,:,:,:) ;
        T1r = permute(T1r,[4,3,2,1]) ;
        T1savename = ['./SamplingForBrainStructureAwareNetwork/train-1-T1r'] ;
        save(T1savename, 'T1r', '-v7.3')
        clear IIin1 T1r
        
        T2r = IIin2(list,:,:,:) ;
        T2r = permute(T2r,[4,3,2,1]) ;
        T2savename = ['./SamplingForBrainStructureAwareNetwork/train-1-T2r'] ;
        save(T2savename, 'T2r', '-v7.3')
        clear IIin2 T2r

        T3r = IOout(list,:,:,:) ;
        T3r = permute(T3r,[4,3,2,1]) ;
        T3savename = ['./SamplingForBrainStructureAwareNetwork/train-1-T3r'] ;
        save(T3savename, 'T3r', '-v7.3')
        clear Iout T3r

        T4r = IOsegsave(list,:,:,:,:) ;
        T4r = permute(T4r,[5,4,3,2,1]) ;
        T1savename = ['./SamplingForBrainStructureAwareNetwork/train-1-T4r'] ;
        save(T1savename, 'T4r', '-v7.3')
        clear IOsegsave T4r
    end
    
    
    
end


