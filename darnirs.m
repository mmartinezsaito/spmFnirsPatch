%% Changelog by mms
%{
  spm_fnirs_read_nirscout.m   : line 84
  spm_fnirs_convert_ui.m      : line 83
  spm_fnirs_temporalpreproc_ui: line 31, 95-7, 120, 123-5
  spm_fnirs_con_2d            : line 11
  spm_fnirs_con_3d            : line 11, 24
%}

%% Naming and settings

clear

% Paths
%devdir = '/home/mario/Data/';
devdir = '/home/mario/';
%bdir1  = [devdir 'MEGA/Neuroscience/Emotion_Interoception_Action_Learning/pj22_fNIRS/'];
bdir1  = [devdir 'Data/pj22_fNIRS_ROI/'];
bdir2 =  [devdir 'MEGA/StatisticalLearning_Probability_InformationTheory/'];
pjname = 'kupimuzyku'  % 'kupimuzyku, bidfood, vkuspolza

% configuration variables
wl = [760 850];  % wavelengths [nm] of the *.wl1 and *.wl2 files
molabs_HbO = [1.4033 3.8547];  % molar absorption coefficients [mM^−1 cm^−1] of oxy-hemoglobin at wavelength 1 and wavelength 2
molabs_HbR = [2.6694 1.8096];  % molar absorption coefficients [mM^−1 cm^−1] of deoxy-hemoglobin
sdd = 3;  % source-detector distance [cm]
%DPF       differential pathlength factor at wavelength 1 and wavelength 2
fs = 3.9062;

% which Hb to read
whichHb = [1 ]; %[1 2 3];
str{1,1} = 'HbO'; str{2,1} = 'HbR'; str{3,1} = 'HbT';
str_nirs = str(whichHb)    % chromophores to be analysed: 'HbO', 'HbR', 'HbT'

% add SPM and SPM-fNIRS to the matlab path
addpath(genpath([bdir2 'Neuroimaging/NIRS_PET/spm_fnirs']))
addpath([bdir2 'spm12'])
addpath([bdir2 'Neuroimaging/Anatomy_Templates_Segmentation_ROI'])

% Data format variables
switch pjname
    case 'kupimuzyku' 
        sids = setdiff(1:40, [1 12 29]); % 1:40, find(mean([E{:,4}],1)==0)
        sdi = 0/fs; % stimulus display interval 
        datcol = 5;
        scanformat = '%f%f%s%u%s'; nhl = 0;
        % LONI: Inferior frontal gyrus 80: AF7(S2) F7(S4) F5(D2)
        % LONI: Middle frontal gyrus 80:   AF3(S3) F3(S5) Fp1(D5) F1(D4) FC3(D3) 
        doroi = 0; rois = [2 3 4 5]; roid = [2 3 4 5];
        subdir2 = 'whole_sdi0/';
    case 'bidfood'
        sids = 4:40;
        sdi = 0/fs;
        datcol = 4;
        scanformat = '%*q%*u%f%q%f%*u%q'; nhl = 1;
        % Brodmann: DLPFC 60: AFz(D7) F1(D4) F2(D9) Fz(S9) F3(S5) F4(S12)
        doroi = 1; rois = [5 9 12]; roid = [4 7 9];
        subdir2 = 'ROI_dlpfc_sdi0_wtp123/';
    case 'vkuspolza'
        sids = 1:39;
        sdi = 0/fs;
        datcol = 7;
        scanformat = '%f%s%s%s%d%u%u'; nhl = 1;
        doroi = 0; %        
        subdir2 = 'whole_sdi0_DecXPvPrimSj/';
end

% Contrast specification
%from line 401


%% File sifting, event reading, regressor specification

ages = dlmread([bdir1 'ages.txt']);
nsid = length(sids);
fof = [pjname filesep '%03u'];
fef = [pjname filesep '%03u.csv']; E = cell(sids(end),datcol);
anadir = {}; 
for s = sids
  names={}; onsets={}; durations={}; pmod={};
  fost = sprintf(fof, s)
  anadir{s} = [bdir1 fost filesep];
  fn1(s) = dir([anadir{s} '*wl1']);
  fn2(s) = dir([anadir{s} '*wl2']);
  fn3(s) = dir([anadir{s} '*hdr']);
  % events reading
  fev = sprintf(fef, s);
  fid = fopen([bdir1 fev]); 
  E(s,:) = textscan(fid, scanformat, 'HeaderLines', nhl, 'Delimiter', ','); %'EndOfLine', '\r\n');
  fclose(fid);
  % writing conditions file 
  if strcmp(pjname, 'kupimuzyku')
	  
    %names{1} = 'purchase';    onsets{1} = E{s,1}(E{s,4}==1);  
    %names{2} = 'no_purchase'; onsets{2} = E{s,1}(E{s,4}~=1);  
    %names{3} = 'Apple';       onsets{3} = E{s,1}(strcmp(E{s,3},'Apple')); 
    %names{4} = 'Spotify';     onsets{4} = E{s,1}(strcmp(E{s,3},'Spotify'));  
    %mnp = mean(E{s,2}); 
    %names{5} = 'price_high'; onsets{5} = E{s,1}(E{s,2}>mnp)      
    %names{6} = 'price_low';  onsets{6} = E{s,1}(E{s,2}<mnp);
    %names{5} = 'stimulus';    onsets{5} = E{s,1};
    %pmod{5} = struct('name', {'price'}, 'P', {zscore(E{s,2})}, 'h', {1});

    names{1} = 'stimulus_buy';   onsets{1} = E{s,1}(E{s,4}==1);
    pmod{1} = struct('name', {'price_buy'}, 'P', {zscore(E{s,2}(E{s,4}==1))}, 'h', {1});
    names{2} = 'stimulus_nobuy'; onsets{2} = E{s,1}(E{s,4}~=1);
    pmod{2} = struct('name', {'price_nobuy'}, 'P', {zscore(E{s,2}(E{s,4}~=1))}, 'h', {1});
    names{3} = 'Apple';          onsets{3} = E{s,1}(strcmp(E{s,3},'Apple')); 
    names{4} = 'Spotify';        onsets{4} = E{s,1}(strcmp(E{s,3},'Spotify'));  

  elseif strcmp(pjname, 'bidfood')
    free_rows = strcmp(E{s,4},'free'); forc_rows = strcmp(E{s,4},'forced');
    %names{1} = 'free'; onsets{1} = E{s,1}(free_rows); 
    %pmod{1} = struct('name', {'free_wtp'}, 'P', {zscore(E{s,3}(free_rows))}, 'h', {1});
    %names{2} = 'forc'; onsets{2} = E{s,1}(forc_rows); 
    %pmod{2} = struct('name', {'forc_wtp'}, 'P', {zscore(E{s,3}(forc_rows))}, 'h', {1});
    %nr0 = length(names);
    %items = unique(str2double(E{s,2})); nitems=length(items);
    %for i=items'
    %  names{nr0+i} = sprintf('item%02u_free',i); onsets{nr0+i} = E{s,1}(str2double(E{s,2})==i & strcmp(E{s,4},'free'));
    %  names{nr0+nitems+i} = sprintf('item%02u_forc',i); onsets{nr0+nitems+i} = E{s,1}(str2double(E{s,2})==i & strcmp(E{s,4},'forced')); %pmod{nr0+nitems+i} = struct('name', {[forcin '_wtp']}, 'P', {(E{s,3}(forcrw)-mu)/sg}, 'h', {1});
    %end
    names{1} = 'free_low';  onsets{1} = E{s,1}(free_rows & E{s,3} < 25); 
    names{2} = 'forc_low';  onsets{2} = E{s,1}(forc_rows & E{s,3} < 25); 
    names{3} = 'free_mid';  onsets{3} = E{s,1}(free_rows & E{s,3} > 25 & E{s,3} < 50); 
    names{4} = 'forc_mid';  onsets{4} = E{s,1}(forc_rows & E{s,3} > 25 & E{s,3} < 50); 
    names{5} = 'free_high'; onsets{5} = E{s,1}(free_rows & E{s,3} > 50);
    names{6} = 'forc_high'; onsets{6} = E{s,1}(forc_rows & E{s,3} > 50);  
    
  elseif strcmp(pjname, 'vkuspolza')
    % frame, product, polezvred, priming, decision, vkus, polza
    %{
    names{1} = 'stimulus';    onsets{1} = E{s,1};
    %pmod{1}(1) = struct('name', {'decision'}, 'P', {zscore(double(E{s,5}))}, 'h', {1});
    pmod{1}(1) = struct('name', {'sjvkus'}, 'P', {zscore(double(E{s,6}))}, 'h', {1});
    pmod{1}(2) = struct('name', {'sjpolz'}, 'P', {zscore(double(E{s,7}))}, 'h', {1});
    
    names{2} = 'polez';   onsets{2} = E{s,1}(strcmp(E{s,3},'polezny'));
    names{3} = 'vred';    onsets{3} = E{s,1}(strcmp(E{s,3},'vred')); 
    
    names{4} = 'primvkus'; onsets{4} = E{s,1}(strcmp(E{s,4},'Отдохните и затем подумайте о ВКУСЕ следующего товара')); 
    names{5} = 'primpolz'; onsets{5} = E{s,1}(strcmp(E{s,4},'Отдохните и затем подумайте о ПОЛЬЗЕ следующего товара')); 
    names{6} = 'primtov';  onsets{6} = E{s,1}(strcmp(E{s,4},'Отдохните и затем просто подумайте о следующем товаре')); 
    %}
    
    names{1} = 'stimulus_eat'; onsets{1} = E{s,1}(E{s,5} > 0);
    pmod{1}(1) = struct('name', {'sjvkus_eat'}, 'P', {zscore(double(E{s,6}(E{s,5}>0)))}, 'h', {1});
    pmod{1}(2) = struct('name', {'sjpolz_eat'}, 'P', {zscore(double(E{s,7}(E{s,5}>0)))}, 'h', {1});
    names{2} = 'polez_eat';    onsets{2} = E{s,1}(strcmp(E{s,3},'polezny') & E{s,5}>0);
    names{3} = 'vred_eat';     onsets{3} = E{s,1}(strcmp(E{s,3},'vred') & E{s,5}>0); 
    names{4} = 'primvkus_eat'; onsets{4} = E{s,1}(strcmp(E{s,4},'Отдохните и затем подумайте о ВКУСЕ следующего товара') & E{s,5}>0); 
    names{5} = 'primpolz_eat'; onsets{5} = E{s,1}(strcmp(E{s,4},'Отдохните и затем подумайте о ПОЛЬЗЕ следующего товара') & E{s,5}>0); 
    names{6} = 'primtov_eat';  onsets{6} = E{s,1}(strcmp(E{s,4},'Отдохните и затем просто подумайте о следующем товаре') & E{s,5}>0); 
   
    names{7} = 'stimulus_not'; onsets{7} = E{s,1}(E{s,5} < 0);
    pmod{7}(1) = struct('name', {'sjvkus_not'}, 'P', {zscore(double(E{s,6}(E{s,5}<0)))}, 'h', {1});
    pmod{7}(2) = struct('name', {'sjpolz_not'}, 'P', {zscore(double(E{s,7}(E{s,5}<0)))}, 'h', {1});
    names{8} = 'polez_not';    onsets{8} = E{s,1}(strcmp(E{s,3},'polezny') & E{s,5}<0);
    names{9} = 'vred_not';     onsets{9} = E{s,1}(strcmp(E{s,3},'vred') & E{s,5}<0); 
    names{10} = 'primvkus_not'; onsets{10} = E{s,1}(strcmp(E{s,4},'Отдохните и затем подумайте о ВКУСЕ следующего товара') & E{s,5}<0); 
    names{11} = 'primpolz_not'; onsets{11} = E{s,1}(strcmp(E{s,4},'Отдохните и затем подумайте о ПОЛЬЗЕ следующего товара') & E{s,5}<0); 
    names{12} = 'primtov_not';  onsets{12} = E{s,1}(strcmp(E{s,4},'Отдохните и затем просто подумайте о следующем товаре') & E{s,5}<0); 
       
  end
  durations = num2cell(ones(1,length(names))*sdi);
  if length(pmod) < length(names), pmod{length(names)} = []; end
  save([anadir{s} 'multiple_conditions.mat'], 'names', 'onsets', 'durations', 'pmod')
end



%% First level analysis

% Start up GUI: spm_fnirs

for s = sids
s, tic, clear SPM

F = {};
F{1,1} = fullfile(anadir{s}, fn1(s).name);
F{2,1} = fullfile(anadir{s}, fn2(s).name); 
F{3,1} = fullfile(anadir{s}, fn3(s).name); F


%
%% Conversion of NIRScout system light intensity to SPM-fNIRS data format
spm_fnirs_read_nirscout(F);
% data will be written to NIRS.mat file ([y P]= wouldn't save matfile)
% stimulus events and channel configuration are read from *.hdr file and 
%  written in ch_config.txt and multiple_conditions.mat files, respectively
% GUI: type "spm_fnirs_read_nirscout" and select NIRScout files (*.wl1, *.wl2, *.hdr)
% output:  y - Light intensity [# samples x # channels x # waves]
%          P - information about raw data structure
%            P.fs       sampling frequency
%            P.nch      number of channels
%            P.ns       number of samples
%            P.ch_mask  mask of channels
%            P.fname    names of raw files and converted file (NIRS.mat) 


%% Conversion to HbO-HbR concentration changes
%  convert fNIRS data to optical density and hemoglobin changes 
load([anadir{s} 'NIRS.mat'])
% y - name of fNIRS files (.txt; .mat) or matrix of measurments (y)
% P - information about data structure and parameters for the modified Beer-Lambert law 
P.age = ages(s, 2);
P.wav = wl;   % light wavelengthsa
% P.mask       mask of measurements
P.d = sdd;    % distance between source and detector 
P.acoef = [molabs_HbO; molabs_HbR]; % molar absorption coefficients [1/(mM*cm)] 
DPF = 223.3 + 0.05624*(P.age^0.8493) + (-5.723) * (10^(-7)) .* (P.wav.^3) + 0.001245 .* (P.wav.^2) + (-0.9025) .* P.wav;
P.dpf = DPF;  % differential pathlength factor [unitless] 
% P.base       baseline period [scan]
spm_fnirs_convert_ui(y, P) % [Y P]= precludes saving matfile
% GUI: press 'Convert' and follow instructions
% output:  Y - structure array of fNIRS data 
%          P - information about data structure and parameters for the modified Beer-Lambert law 


%% Spatial preprocessing
%
% read estimated cortical projections of optodes on head skin
cd(bdir1)
load Standard_probeInfo.mat
% read NIRscout montage
setup_labels = {'Fp1','Fp2','Fz','F3','F4','F7','F8','Cz','C3','C4','T3','T4','Pz','P3','P4','T5','T6','O1','O2'};
ch_sd = probeInfo.probes.index_c;
labels_d = probeInfo.probes.labels_d; nd = length(labels_d);
labels_s = probeInfo.probes.labels_s; ns = length(labels_s);
labels_sd = [labels_s labels_d];
%{
% Okamoto, M., Dan, H., Sakamoto, K., Takeo, K., Shimizu, K., Kohno, S., … Dan, I. (2004). Three-dimensional probabilistic anatomical cranio-cerebral correlation via the international 10-20 system oriented for transcranial functional brain mapping. NeuroImage, 21(1), 99–111.
H = csvread('Okamoto_Eeg10-20toHeadSurface_MNI.csv', 1, 1, [1 1 19 3]);
CO = dlmread('Okamoto_Eeg10-20toCerebralSurface_MNI.tsv','\t', 1, 1, [1 1 19 3]);
% 10-10MCN: T7:T3, T4:T8, T5:P7, T6:P8
% Koessler, L., Maillard, L., Benhadid, A., Vignal, J. P., Felblinger, J., Vespignani, H., & Braun, M. (2009). Automated cortical projection of EEG sensors: Anatomical correlation via the international 10-10 system. NeuroImage, 46(1), 64–72.
C = csvread('Koessler_Eeg10-10toCorticalSurface_Tal.csv', 1, 1);
% plot optodes and projections
M = csvread([bdir2 'Neuroimaging/NIRS_PET/spm_fnirs/stroop/pos/optode_positions_MNI.csv'],2,1)
scatter3(M(:,1),M(:,2),M(:,3),'h'), hold on
M = csvread([bdir 'optode_positions_vladkos_MNI.txt'],2,1)
scatter3(M(:,1),M(:,2),M(:,3),'.')
M = csvread([bdir 'optode_positions_probeInfo.csv'],2,1)
scatter3(M(:,1),M(:,2),M(:,3),'o')
scatter3(H(:,1),H(:,2),H(:,3),'v')
scatter3(C(:,1),C(:,2),C(:,3),'^')
O = csvread([bdir 'optode_positions_Koessler09.csv'],2,1)
scatter3(O(:,1),O(:,2),O(:,3),'*')
% read Koessler09's projections into K
fid = fopen('Koessler_Eeg10-10toCorticalSurface_Tal.csv');
fgetl(fid)
K = textscan(fid,'%s%f%f%f','Delimiter',',','EndOfLine','\n');
fclose(fid)
K{1}'
% semijoin of current montage and Koessler09's projection MNI coordinates
fid = fopen('optode_positions_Koessler09.csv', 'w'); 
fprintf(fid, 'Optode (MNI),X,Y,Z\n');
for i = 1:length(labels_sd)
  lb = labels_sd{i}
  idx = cellfun(@(x) strcmpi(lb,x), K{1});
  opco = C(idx,:);
  if i <= ns
    fprintf(fid, 'S%u,%3.1f,%3.1f,%3.1f\n', i, opco(1), opco(2), opco(3));
  else
    fprintf(fid, 'D%u,%3.1f,%3.1f,%3.1f\n', i-ns, opco(1), opco(2), opco(3));
  end
end
fclose(fid);
%}
% Transform optode/channel positions from subject space to MNI space 
F = {};
F{1,1}{1,1} = [bdir1 'optode_positions_Koessler09.csv']; 
F{1,1}{2,1} = fullfile(anadir{s}, 'ch_config.txt');
F{1,1} = char(F{1,1});   % it only accepts char array, not cell array
F{2,1} = fullfile(anadir{s}, 'NIRS.mat');
spm_fnirs_spatialpreproc_ui(F) % R= precludes saving 
% GUI: press 'Spatial' and follow instructions
% output: R - structure array of optode/channel positions

%
%% ROI

if doroi
  load([anadir{s} 'NIRS.mat'])
  load(P.fname.pos)
  roisch = ismember(ch_sd(:,1), rois);
  roidch = ismember(ch_sd(:,2), roid);
  roichor = double(or(roisch, roidch)');
  roichand = double(and(roisch, roidch)');

% Using any channel including any ROI source or detector (or)
%  as opposed to including all ROI sources and detectors (and)
  R.ch.mask = roichor; save(P.fname.pos, 'R')  
  P.mask = roichor;    save(P.fname.nirs, 'Y', 'P')
end

%% Temporal preprocessing
% identify channels of interest 
if isfield(P.fname, 'pos') 
    load(P.fname.pos);
    mask = zeros(1, R.ch.nch); indx = find(R.ch.mask == 1);
    mask(R.ch.label(indx)) = 1; clear R;
else, mask = ones(1, P.nch); 
end
mask = mask .* P.mask; ch_roi = find(mask ~= 0); 
% Apply temporal filters to fNIRS data 
load(fullfile(anadir{s}, 'NIRS.mat'))
P.K.M.type = 'MARA'; % motion artifact correction
P.K.M.chs = ch_roi;
P.K.M.L = 1;   
P.K.M.th = 3;  
P.K.M.alpha = 5; 
P.K.C.type = 'Band-stop filter';
P.K.C.cutoff = [0.12 0.35; 0.7 1.5];
P.K.D.type = 'yes'; % donwsampling 
P.K.D.nfs = 1;   
P.K.H.type = 'DCT'; % detrending
P.K.H.cutoff = 128;    
P.K.L.type = 'no';  % temporal smoothing
save(P.fname.nirs, 'P', '-append', spm_get_defaults('mat.format'))
spm_fnirs_temporalpreproc_ui(fullfile(anadir{s}, 'NIRS.mat'))
% GUI: press 'Temporal' and follow instructions
% output: P - structure array of filter parameters (P.K) 

%}
%% Model specification
% Specify general linear model (GLM) of fNIRS for the 1st level analysis 
% E: frame, price, brand, purchase:1, responsebutton
load(fullfile(anadir{s}, 'NIRS.mat'))
% anadir{s};         % directory to save SPM.mat file
% specify experimental design
SPM.xBF.UNITS = 'scans';  % time units in 'scans' ('frames' for fNIRS) or 'secs'
load(fullfile(anadir{s},'multiple_conditions.mat')), names, durations, onsets, pmod    
for i = 1:size(names, 2)
    U(i).name = names(1,i);
    U(i).ons = onsets{1,i}(:);
    U(i).dur = durations{1,i}(:); 
end
% parametric modulations
for i = 1:numel(U)
    if isempty(pmod{i})
        U(i).P.name = 'none';
        U(i).P.h = 0;
    else % see spm_get_ons for designed effects structure  
        % U(i).orth = 1;
        U(i).P = pmod{i};
    end
end
% resampling
switch P.K.D.type
    case 'yes', rtfs = P.K.D.nfs; nscan = P.K.D.ns;
                for i = 1:size(names, 2), U(i).ons = U(i).ons / fs; end
    case 'no',  rtfs = P.fs;      nscan = P.ns;     
end
SPM.xY.RT = 1/rtfs;
SPM.nscan = nscan;
SPM.Sess.U = U;
% basis functions
try   SPM.xBF.T  = spm_get_defaults('stats.fmri.t');
catch SPM.xBF.T = 16;
end
try   SPM.xBF.T0 = spm_get_defaults('stats.fmri.t0');
catch SPM.xBF.T0 = 8;
end
SPM.xBF.dt = SPM.xY.RT/SPM.xBF.T;
SPM.xBF.Volterra = 1; % 1st Volterra expansion
SPM.Sess.U = spm_get_ons(SPM,1);
% covariates
C = []; c = 0;  % C is a matrix with covariates as columns of length ns(cans)
Cname = cell(1,size(C,2));
SPM.Sess.C.C = C;
SPM.Sess.C.name = Cname;
% Intrinsic autocorrelations (Vi) for non-sphericity ReML estimation
SPM.xVi.Vi = spm_Ce(nscan,0.2); % assume AR(0.2) in xVi.Vi
SPM.xVi.form = 'AR(0.2)';
% specify to run spm_get_bf without prompts
%  'hrf', 'hrf (with time derivative)', 'hrf (with time and dispersion derivatives)'
%  'Fourier set', 'Fourier set (Hanning)', 'Gamma functions', 'Finite Impulse Response'
SPM.xBF.name = 'hrf';  
% generate design matrix using specified parameters
SPM.xY.VY = P.fname.nirs; % file name of Y
SPM0 = SPM; save_SPM = 0;
for i = 1:size(str_nirs, 1)
    [SPM] = spm_fMRI_design(SPM0, save_SPM);
    SPM.xY.type = str_nirs{i,1}; % hemoglobin type (eg, HbO or HbR)
    if strcmpi(str_nirs{i,1}, 'HbR')
        if strcmpi(SPM.xBF.name, 'hrf') || strcmpi(SPM.xBF.name, 'hrf (with time derivative)') || strcmpi(SPM.xBF.name, 'hrf (with time and dispersion derivatives)')
            SPM.xX.X(:,1:end-1) = SPM.xX.X(:,1:end-1).*(-1);
        end
    end   
    %-Design description - for saving and display
    ntr = length(SPM.Sess.U);
    Bstr = sprintf('[%s] %s', str_nirs{i,1}, SPM.xBF.name);
    Hstr = P.K.H.type; 
    if strcmpi(Hstr, 'DCT'), Hstr = sprintf('%s, Cutoff: %d {s}', Hstr, P.K.H.cutoff); end 
    Lstr = P.K.L.type;
    if strcmpi(Lstr, 'Gaussian'), Lstr = sprintf('%s, FWHM %d', Lstr, P.K.L.fwhm); end
    SPM.xsDes = struct(...
        'Basis_functions',      Bstr,...
        'Number_of_sessions',   sprintf('%d',1),...
        'Trials_per_session',   sprintf('%-3d',ntr),...
        'Interscan_interval',   sprintf('%0.2f {s}',SPM.xY.RT),...
        'High_pass_Filter',     Hstr,...
        'Low_pass_Filter', Lstr);
    % for display
    spm_DesRep('DesMtx',SPM.xX,[],SPM.xsDes);
    % output directory
    swd = fullfile(anadir{s}, str_nirs{i,1});
    if ~isdir(swd), mkdir(swd), end
    SPM.swd = swd;
    % saving 
    save(fullfile(SPM.swd, 'SPM.mat'), 'SPM', spm_get_defaults('mat.format'));
    % Delete files related to previous SPM.mat file (if exist)
    fname = spm_select('FPList', SPM.swd, '^con_.*\.mat$');
    if ~isempty(fname)
        for i = 1:size(fname, 1), delete(deblank(fname(i,:))); end
    end    
    fname = spm_select('FPList', SPM.swd, '^spmT_.*\.mat$');
    if ~isempty(fname)
        for i = 1:size(fname, 1), delete(deblank(fname(i,:))); end
    end    
    fname = spm_select('FPList', SPM.swd, '^spmF_.*\.mat$');
    if ~isempty(fname)
        for i = 1:size(fname, 1), delete(deblank(fname(i,:))); end
    end
end
% GUI : press 'Specify 1st level' and follow instructions


%% Model estimation
% Estimate GLM parameters for SPM analysis of fNIRS data for each chromophore
F = {};
for i = 1:size(str_nirs, 1), str_nirs{i}
    F{i} = fullfile(anadir{s},str_nirs{i},'SPM.mat');
    spm_fnirs_spm(F{i}); 
end


%% Results
% GUI: spm_fnirs_results_ui(SPM)
% [STATmode,n,Prompt,Mcstr,OK2chg] = deal('T&F',Inf,'    Select contrasts...',' for conjunction',1);
% Interpolate GLM parameters
for i = 1:size(str_nirs, 1)
  load(fullfile(anadir{s},str_nirs{i},'SPM.mat'))
% Interpolate GLM parameters over surfaces of rendered brain 
  if ~isfield(SPM, 'Vbeta') || ~isfield(SPM, 'VResMS') || ~isfield(SPM.xVol, 'VRpv')  
    [SPM] = spm_fnirs_spm_interp(SPM);
  end
  
% Specify contrast vector, compute inference SPM and
% visualize thresholded SPM on a surface of rendered brain template 
% GUI: spm_fnirs_viewer_stat(SPM)
%  (i)   contrast vector is specified, using spm_conman.m 
%  (ii)  inference SPM is computed, using spm_fnirs_contrasts.m 
%  (iii) height threshold is computed, using spm_uc.m 
% Commands to find spm_FcUtil
%  in spm_conman at 1332, findobj(F,'Tag','D_ConMtx')
%  dbclear all
%  dbstop in spm_conman at 527 if (exist('F') && ~isempty(get(findobj(F,'Tag','D_ConMtx'),'UserData')))
%  dbstatus
%  [ic,xCon] = spm_conman(SPM,'T&F',Inf,'    Select contrasts...',' for conjunction',1);
% Contrasts are stored by SPM in a single structure 
%  see spm_FcUtil.m for its definition and handling 
%  spm_FcUtil('FconFields') % return fields of contrast structure
% The xCon structure for each contrast contains data specific to the current 
%  experimental design, so contrast structures can only be copied between 
%  analyses (to save re-entering contrasts) if the designs are *identical*
% contrasts are set in spm_conman at 1264
  namec = {}; STATc = {}; cc = {};
  if strcmp(pjname, 'kupimuzyku')    % [1 1 0 0 -1 0 0] and [0 0 1 1 -1 0 0] are linearly dependent directions
    namec{1} = 'buy-nobuy';       STATc{1} = 'T'; cc{1} = [1 0 -1 0 0 0 0]'; 
    namec{2} = 'Apple-Spotify';   STATc{2} = 'T'; cc{2} = [0 0 0 0 1 -1 0]'; 
    namec{3} = 'stimulus';        STATc{3} = 'T'; cc{3} = [1 0 1 0 0 0 0]'; 
    namec{4} = 'price';           STATc{4} = 'T'; cc{4} = [0 1 0 1 0 0 0]'; 
    
    namec{5} = 'price*buy-nobuy'; STATc{5} = 'T'; cc{5} = [0 1 0 -1 0 1 0]';
    
    
  elseif strcmp(pjname, 'bidfood')
    %namec{1} = 'free-forc';  STATc{1} = 'T'; cc{1} = [1 -1 0 0 0]';
    %namec{2} = 'stim-dec';   STATc{2} = 'T'; cc{2} = [0 0 1 0 0]'; 
    %namec{3} = 'free_wtp';   STATc{3} = 'T'; cc{2} = [1 0 0 1 0]'; 
    %namec{4} = 'forced_wtp'; STATc{4} = 'T'; cc{3} = [0 1 0 1 0]'; 
    
    %namec{1} = 'wtp';          STATc{1} = 'T'; cc{1} = [0 1  0 1 zeros(1,2*nitems+1)]'; 
    %namec{2} = 'free_wtp';     STATc{2} = 'T'; cc{2} = [0 1  0 0 zeros(1,2*nitems+1)]'; 
    %namec{3} = 'forc_wtp';     STATc{3} = 'T'; cc{3} = [0 0  0 1 zeros(1,2*nitems+1)]'; 
    %namec{4} = 'free-forc';    STATc{4} = 'T'; cc{4} = [1 0 -1 0 zeros(1,2*nitems+1)]';
    %namec{5} = 'free-forc_wtp';STATc{5} = 'T'; cc{5} = [0 1 0 -1 zeros(1,2*nitems+1)]';
   
    %nc0 = length(namec); cc0 = zeros(5+2*nitems,1);
    %cc0 = zeros(2*2*nitems+1, 1);
    %namec{1} = 'wtp';          STATc{1} = 'T'; cc{1} = cc0; cc{1}(2:2:4*nitems) = 1; 
    %namec{2} = 'free_wtp';     STATc{2} = 'T'; cc{2} = cc0; cc{2}(2:2:2*nitems) = 1; 
    %namec{3} = 'forc_wtp';     STATc{3} = 'T'; cc{3} = cc0; cc{3}(2*nitems+2:2:4*nitems) = 1; 
    %namec{4} = 'free-forc';    STATc{4} = 'T'; cc{4} = cc0; cc{4}(1:2:2*nitems) = 1; cc{4}(2*nitems+1:2:4*nitems) = -1; 
    %namec{5} = 'free-forc_wtp';STATc{5} = 'T'; cc{5} = cc0; cc{5} = cc{2} - cc{3};
    %for j=items'
       %namec{nc0+j} = sprintf('free_item%02u',j); STATc{nc0+j}='T'; cc{nc0+j}=cc0; cc{nc0+j}(4+j) = 1;     
       %namec{nc0+nitems+j} = sprintf('forc_item%02u',j); STATc{nc0+nitems+j}='T'; cc{nc0+nitems+j}=cc0; cc{nc0+nitems+j}(4+nitems+j) = 1; 
      %namec{nc0+j} = sprintf('free-forc_item%02u',j); STATc{nc0+j}='T'; cc{nc0+j}=cc0; cc{nc0+j}([4+j,4+nitems+j]) = [1 -1];     
      
       %namec{nc0+j} = sprintf('free-forc_item%02u',j); STATc{nc0+j}='T';
       %cc{nc0+j}=cc0; cc{nc0+j}([2*j-1,2*nitems+2*j-1]) = [1 -1];     
       %namec{nc0+nitems+j} = sprintf('free-forc_item%02u_wtp',j); STATc{nc0+nitems+j}='T';
       %cc{nc0+nitems+j}=cc0; cc{nc0+nitems+j}([2*j,2*nitems+2*j]) = [1 -1];     
    %end
    
    %namec{1} = 'free';      STATc{1} = 'T'; cc{1} = [1 0 1 0 1 0 0]'; 
    %namec{2} = 'forc';      STATc{2} = 'T'; cc{2} = [0 1 0 1 0 1 0]'; 
    %namec{3} = 'low';       STATc{3} = 'T'; cc{3} = [1 1 0 0 0 0 0]'; 
    %namec{4} = 'mid';       STATc{4} = 'T'; cc{4} = [0 0 1 1 0 0 0]'; 
    %namec{5} = 'high';      STATc{5} = 'T'; cc{5} = [0 0 0 0 1 1 0]'; 
    namec{1} = 'free-forc';     STATc{1} = 'T'; cc{1} = [1 -1 1 -1 1 -1 0]'; 
    namec{2} = 'free_high-low'; STATc{2} = 'T'; cc{2} = [-1 0 0 0 1 0 0]';
    namec{3} = 'free_high-mid'; STATc{3} = 'T'; cc{3} = [0 0 -1 0 1 0 0]';
    namec{4} = 'free_mid-low';  STATc{4} = 'T'; cc{4} = [-1 0 1 0 0 0 0]'; 
    namec{5} = 'forc_high-low'; STATc{5} = 'T'; cc{5} = [0 -1 0 0 0 1 0]';
    namec{6} = 'forc_high-mid'; STATc{6} = 'T'; cc{6} = [0 0 0 -1 0 1 0]';
    namec{7} = 'forc_mid-low';  STATc{7} = 'T'; cc{7} = [0 -1 0 1 0 0 0]'; 
    namec{8} = 'high_free-forc'; STATc{8} = 'T';  cc{8} = [0 0 0 0 1 -1 0]'; 
    namec{9} = 'mid_free-forc';  STATc{9} = 'T';  cc{9} = [0 0 1 -1 0 0 0]'; 
    namec{10} = 'low_free-forc'; STATc{10} = 'T'; cc{10} = [1 -1 0 0 0 0 0]'; 
    namec{11} = 'free-forc*high-low'; STATc{11} = 'T'; cc{11} = [-1 1 0 0 1 -1 0]'; 
           
  elseif strcmp(pjname, 'vkuspolza')
    %namec{1} = 'decision';   STATc{1} = 'T'; cc{1} = [0 1 0 0 0]'; 
    
    %namec{2} = 'polez-vred'; STATc{2} = 'T'; cc{2} = [0 0 1 -1 0]'; 
    %namec{3} = 'posdec*polez-vred'; STATc{3} = 'T'; cc{3} = [0 1 1/2 -1/2 0]';      
    %namec{4} = 'negdec*polez-vred'; STATc{4} = 'T'; cc{4} = [0 -1 1/2 -1/2 0]';
    
    %namec{2} = 'primVkus';        STATc{2} = 'T'; cc{2} = [0 0 1 0 -1 0]'; 
    %namec{3} = 'primPolz';        STATc{3} = 'T'; cc{3} = [0 0 0 1 -1 0]';    
    %namec{4} = 'dec*primVkus';    STATc{4} = 'T'; cc{4} = [0 1 1/2 0 -1/2 0]';      
    %namec{5} = 'negdec*primVkus'; STATc{5} = 'T'; cc{5} = [0 -1 1/2 0 -1/2 0]';      
    %namec{6} = 'dec*primPolz';    STATc{6} = 'T'; cc{6} = [0 1 0 1/2 -1/2 0]';      
    %namec{7} = 'negdec*primPolz'; STATc{7} = 'T'; cc{7} = [0 -1 0 1/2 -1/2 0]';  

    %namec{2} = 'sjvkus'; STATc{2} = 'T'; cc{2} = [0 0 1 0 0]'; 
    %namec{3} = 'sjpolz'; STATc{3} = 'T'; cc{3} = [0 0 0 1 0]'; 
    %namec{4} = 'dec*sjvkus'; STATc{4} = 'T'; cc{4} = [0 1/2 1/2 0 0]'; 
    %namec{5} = 'dec*sjpolz'; STATc{5} = 'T'; cc{5} = [0 1/2 0 1/2 0]'; 
    
    %namec{1} = 'decision';   STATc{1} = 'T'; cc{1} = [0 1 0 0 0 0 0 0 0 0]'; 
    %namec{2} = 'polez-vred'; STATc{2} = 'T'; cc{2} = [0 0 0 0 1 -1 0 0 0 0]'; 
    %namec{3} = 'primVkus';   STATc{3} = 'T'; cc{3} = [0 0 0 0 0 0 1 0 -1 0]'; 
    %namec{4} = 'primPolz';   STATc{4} = 'T'; cc{4} = [0 0 0 0 0 0 0 1 -1 0]'; 
    %namec{5} = 'sjvkus';     STATc{5} = 'T'; cc{5} = [0 0 1 0 0 0 0 0 0 0]'; 
    %namec{6} = 'sjpolz';     STATc{6} = 'T'; cc{6} = [0 0 0 1 0 0 0 0 0 0]';  
    
    namec{1} = 'eat-not';    STATc{1} = 'T'; cc{1} = [1 0 0 0 0 0 0 0   -1 0 0 0 0 0 0 0 0]'; 
    namec{2} = 'polez-vred'; STATc{2} = 'T'; cc{2} = [0 0 0 1 -1 0 0 0  0 0 0 1 -1 0 0 0 0]'; 
    namec{3} = 'primVkus';   STATc{3} = 'T'; cc{3} = [0 0 0 0 0 1 0 -1  0 0 0 0 0 1 0 -1 0]'; 
    namec{4} = 'primPolz';   STATc{4} = 'T'; cc{4} = [0 0 0 0 0 0 1 -1  0 0 0 0 0 0 1 -1 0]';  
    namec{5} = 'sjvkus';     STATc{5} = 'T'; cc{5} = [0 1 0 0 0 0 0 0   0 1 0 0 0 0 0 0  0]'; 
    namec{6} = 'sjpolz';     STATc{6} = 'T'; cc{6} = [0 0 1 0 0 0 0 0   0 0 1 0 0 0 0 0  0]'; 
    namec{7} = 'eat-not*plz-vrd';  STATc{7} = 'T';  cc{7} =  [0 0 0 1 -1 0 0 0  0 0 0 -1 1 0 0 0 0]'; 
    namec{8} = 'eat-not*primVkus'; STATc{8} = 'T';  cc{8} =  [0 0 0 0 0 1 0 -1  0 0 0 0 0 -1 0 1 0]'; 
    namec{9} = 'eat-not*primPolz'; STATc{9} = 'T';  cc{9} = [0 0 0 0 0 0 1 -1  0 0 0 0 0 0 -1 1 0]';  
    namec{10} = 'eat-not*sjvkus';  STATc{10} = 'T'; cc{10} = [0 1 0 0 0 0 0 0   0 -1 0 0 0 0 0 0 0]'; 
    namec{11} = 'eat-not*sjpolz';  STATc{11} = 'T'; cc{11} = [0 0 1 0 0 0 0 0   0 0 -1 0 0 0 0 0 0]'; 
    
  end
  
  nc1 = length(namec);
  for j = 1:nc1 % spm_fnirs_viewer_stat at 135
    xCon(j) = spm_FcUtil('Set', namec{j}, STATc{j}, 'c', cc{j}, SPM.xX.xKXs);
    SPM.xCon = xCon;
    SPM = spm_fnirs_contrasts(SPM, j);
  end
end

% To view first level images: spm_fnirs_viewer_stat(SPM)
%}

%% Compute individual nii and gii contrast images
% nii: on 2D regular grid;  gii: on 3D triangular mesh

fname = {}; 
for i = 1:size(str_nirs, 1)
  for j = 1:nc1
    confn = sprintf('con_%04u.mat',j);    
    fname{j,i} = fullfile(anadir{s}, str_nirs{i}, confn);    
  end  
end

% Compute more images if repeated measures design
%  validated by using spm_imcalc (see right below)
if strcmp(pjname, 'bidfood'), S={};
  for i = 1:size(str_nirs, 1), Sj={}; 
    for j = 1:nc1, Sj{j}=load(fname{j,i}); end
    S = Sj{j}.S; 
    S.cbeta = mean(cell2mat(cellfun(@(S) S.S.cbeta, Sj,'Un',0)'));
    j = j+1; confn = sprintf('con_%04u.mat',j);
    fname{j,i} = fullfile(anadir{s}, str_nirs{i}, confn);
    save(fname{j,i}, 'S')
  end
end

fname = char(fname);

spm_fnirs_con_2d 
spm_fnirs_con_3d

% Compute more images if repeated measures design 
%{
for i = 1:size(str_nirs, 1)
  if strcmp(pjname, 'bidfood')
    fname = {}; 
    for j = 1:nc1
      confn = sprintf('con_%04u.nii',j); fname{j} = fullfile(anadir{s}, str_nirs{i}, confn);
    end, fname = char(fname);
    Vi = spm_vol(fname(nc0+1:nc0+nitems,:));
    Vo = fullfile(anadir{s}, str_nirs{i}, sprintf('con_%04u.nii', nc1+2));
    Vo = spm_imcalc(Vi, Vo, 'mean(X)', {1}); %mean(X(1:30,:)-X(31:60,:)) % help spm_imcalc
    
    %Vi = spm_vol(fname(nc0+nitems+1:nc1,:));
    %Vo = fullfile(anadir{s}, str_nirs{i}, sprintf('con_%04u.nii',1000));
    %Vo = spm_imcalc(Vi, Vo, 'mean(X)', {1});     
  end
end  
%}

s, toc
end




%% Second level analysis
clear condir

gldir = [bdir1 pjname filesep 'secondLevel/' subdir2]
conimg = dir([anadir{sids(1)} 'HbO/con_*nii']);
consur = dir([anadir{sids(1)} 'HbO/con_*gii']);

for i=1:size(str_nirs, 1)
  % model specification
  ghdir = [gldir str_nirs{i} filesep]; str_nirs{i}
  for j=1:length(consur)   %  1:3
    % Create dirs if non-existent
    condir{1} = ['2dCon' sscanf(conimg(j).name, 'con_%4s.nii')]; % 1sTtest images
    if exist([ghdir condir{1}], 'file') ~= 7, mkdir(ghdir, condir{1}); end
    %condir = ['2dCon' sscanf(conimg(j).name, 'con_%4s.nii') '_aov']; % 1wAnova
    %if exist([ghdir condir], 'file') ~= 7, mkdir(ghdir, condir); end
    condir{2} = ['3dCon' sscanf(consur(j).name, 'con_%4s.gii')]; % 1sTtest surfaces 
    if exist([ghdir condir{2}], 'file') ~= 7, mkdir(ghdir, condir{2}); end    
    % Fill parameters
    F = []; fi = 0; 
    for s = sids
      fi = fi + 1;
      F{1}{fi} = [anadir{s} str_nirs{i} filesep conimg(j).name];
      F{2}{fi} = [anadir{s} str_nirs{i} filesep consur(j).name];
    end
    for k = 1:2
      clear matlabbatch  %,load([gldir num2str(k+1) 'dGroupAnalysisJob.mat'])
      matlabbatch{1}.spm.stats.factorial_design.dir = {[ghdir condir{k}]};
      matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = F{k}';
      output_list = spm_jobman('run', matlabbatch);      
  % model estimation
      clear matlabbatch  %,load([gldir 'estimate_i.mat'])
      matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;        
      matlabbatch{1}.spm.stats.fmri_est.spmmat = {[ghdir condir{k} filesep 'SPM.mat']};
      output_list = spm_jobman('run', matlabbatch);
    end       
  end
end
