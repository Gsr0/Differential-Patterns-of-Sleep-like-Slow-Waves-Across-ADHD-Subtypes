%%   运行 ICA  
group1_dir = 'D:\桌面\T0_ANT数据v2\set';  
group1_dir1 = 'D:\桌面\T0_ANT数据v2\set';    
group1_files   = dir([group1_dir1, filesep, '*.set']);  
for i=1:length(group1_files)
    subj_fn = group1_files(i).name;
    EEG = pop_loadset('filename',strcat(subj_fn(1:end-4), '.set'), 'filepath', group1_dir1); %导入数据
    EEG = pop_runica(EEG, 'icatype', 'runica','extended', 1, 'interrupt', 'on');   % 跑ICA
    EEG = pop_saveset( EEG, 'filename',strcat(group1_files(i).name(1:end-4), '.set'), 'filepath',strcat(group1_dir, filesep, 'ICA'));  %保存数据
    
end
%% 使用ICLabel自动去除ICA成分
group1_dir = 'D:\桌面\T0_ANT数据v2\ICA后未分段全部数据';     
group1_dir2 = 'D:\桌面\T0_ANT数据v2\ICA后未分段全部数据';  
group1_files = dir([group1_dir2, filesep, '*.set']);  
for i=1:length(group1_files)
    subj_fn = group1_files(i).name;
    EEG = pop_loadset('filename',strcat(subj_fn(1:end-4), '.set'), 'filepath', group1_dir2);
    EEG = pop_iclabel(EEG, 'default');
    EEG = pop_icflag(EEG, [NaN NaN;0.9 1;0.9 1;NaN NaN;NaN NaN;NaN NaN;NaN NaN]); % 标记伪迹成分。这里可以自定义设定阈值，依次为Brain, Muscle, Eye, Heart, Line Noise, Channel Noise, Other.
    EEG = pop_subcomp( EEG, [], 0);   %去除上述伪迹成分
    EEG = eeg_checkset( EEG );
    EEG = pop_saveset( EEG, 'filename',strcat(group1_files(i).name(1:end-4), '.set'), 'filepath',strcat(group1_dir, filesep, 'ICA2')); 
end