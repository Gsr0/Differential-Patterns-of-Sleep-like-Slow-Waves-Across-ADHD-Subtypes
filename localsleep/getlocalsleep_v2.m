%% wanderIM_getLocalSleep_cont_v2.m 最终修正版
%%%%% 预处理检查
%%%%% 预处理后的EEG数据分析

%% 初始化
clear; close all; clc;
addpath(genpath('D:\桌面\论文\注意力检测\ADHD\慢波预测注意力缺失\wanderIM-master\localsleep')); % 替换为实际路径
addpath('D:\桌面\论文\注意力检测\ADHD\慢波预测注意力缺失\wanderIM-master\localsleep'); % 包含twalldetectnew_TA_v2.m的路径
addpath('D:\MATLAB-workspace\m')
%% 参数设置
input_dir = 'D:\桌面\T0_ANT数据v2\ICA后未分段全部数据\ICA2'; % .set文件存储路径
output_dir = 'D:\桌面\T0_ANT数据v2\ICA后未分段全部数据\ICA2'; % 结果输出路径
if ~exist(output_dir, 'dir')
    mkdir(output_dir)   
end

%% 参数定义
params_list = {'maxnegpkamp','maxpospkamp',...
    'mxdnslp','mxupslp','maxampwn','minampwn', 'global_sw_density', 'mean_duration'}; % 6个参数
n_params = length(params_list);
channels = 30;

%% ==== 关键修正1：正确初始化表格 ====
% 预先定义所有列名和类型
var_types = [{'string'}, repmat({'double'},1,n_params)];
var_names = ['SubjectID', params_list];
summary_table = table('Size',[0, n_params+1],...
                    'VariableTypes', var_types,...
                    'VariableNames', var_names);

%% 获取文件列表
set_files = dir(fullfile(input_dir, '*.set'));
n_subjects = length(set_files);
fprintf('发现%d个被试数据\n', n_subjects);

%% 批量处理
for subj_idx = 1:n_subjects
    try
        %% 数据加载
        filename = set_files(subj_idx).name;
        [~, subj_id] = fileparts(filename);
        
        %% ==== 关键修正2：添加路径存在性检查 ====
        if ~exist(fullfile(input_dir, filename), 'file')
            error('文件 %s 不存在', filename);
        end
        
        % 使用EEGLAB加载数据
        EEG = pop_loadset('filename', filename, 'filepath', input_dir);
        fprintf('\n处理被试 %d/%d: %s\n', subj_idx, n_subjects, subj_id);

        %% 初始化被试级变量
        subject_param_means = zeros(1, n_params); % 存储所有epoch的平均
        epoch_count = 0; % 有效epoch计数器
        vaild_count = 0;
        count = 0;
        global_sw_density = 0;
        mean_duration = 0;
        for epoch_idx = 1:EEG.trials
            %% 数据预处理
            temp_data = EEG.data(1:channels,:, epoch_idx); % 选择前30个通道
            %temp_data = temp_data - mean(temp_data, 1); % 平均参考

            %% 慢波检测
            [twa_results] = twalldetectnew_TA_v2(temp_data, EEG.srate, 40);
            % 6. 结果保存
            savename = fullfile(output_dir, [subj_id '_slowwave.mat']);
            save(savename, 'twa_results', '-v7.3');

            %% ==== 关键修正3：安全参数提取 ====
            param_means = nan(1, n_params);
            for p = 1:n_params-2
                all_vals = [];
                for nE = 1:channels
                    % 检查字段是否存在
                    if isfield(twa_results.channels(nE), params_list{p})
                        param_data = twa_results.channels(nE).(params_list{p});
                        if iscell(param_data) && ~isempty(param_data)
                            num_data = cell2mat(param_data);
                            all_vals = [all_vals; num_data(:)];
                        end
                    end
                end
                % 计算全局平均
                if ~isempty(all_vals)
                    param_means(p) = mean(all_vals, 'omitnan');
                    subject_param_means(p) = subject_param_means(p) + param_means(p);
                    vaild_count = 1;
                else
                    param_means(p) = NaN;
                    vaild_count = 0;
                end
            end
            count = vaild_count + count;
            global_sw_density =  global_sw_density + twa_results.global_sw_density;
            mean_duration = mean_duration + twa_results.mean_duration;
            epoch_count = epoch_count + 1;
        end
        
        subject_param_means = subject_param_means / count;
        global_sw_density = global_sw_density / epoch_count;
        mean_duration = mean_duration / epoch_count;
        subject_param_means(7) = global_sw_density;
        subject_param_means(8) = mean_duration;
        %% ==== 关键修正4：安全添加数据 ====
        new_row = cell2table([{subj_id}, num2cell(subject_param_means)],...
            'VariableNames', var_names);
        summary_table = [summary_table; new_row];
        
        %% 进度显示
        fprintf('完成进度: %d/%d (%.1f%%)\n',...
            subj_idx, n_subjects, 100*subj_idx/n_subjects);
        
    catch ME
        fprintf(2,'处理被试 %s 失败: %s\n', subj_id, ME.message);
    end
end

%% 保存结果到Excel
output_file = fullfile(output_dir, 'AllSubjects_Summary_v3.xlsx');
writetable(summary_table, output_file, 'Sheet', 'Summary');
fprintf('结果已保存至: %s\n', output_file);


function [twa_results]=twalldetectnew_TA_v2(datainput,orig_fs,thramp)
%% Theta Wave Detection with Slow Wave Density Calculation
% 添加了慢波密度计算功能

disp(['**** Delta/Theta Wave Detection with Density Calculation ****']);

%% 初始化密度字段
twa_results = struct();
twa_results.global_sw_count = 0;
twa_results.global_sw_density = 0;
twa_results.mean_duration = 0;

%% 原有代码保持不变...
allavg=nanmean(datainput,2); 
dataref=datainput-repmat(allavg,[1,size(datainput,2)]);
LoadedEEG.data=dataref; clear dataref;  

%% Filter Definition
fs=128; %sampling rate changes for decimated signal
Wp=[1.0 10.0]/(fs/2); % Filtering parameters
Ws=[0.1 15]/(fs/2); % Filtering parameters
Rp=3;
Rs=25;
[n, Wn]=cheb2ord(Wp,Ws,Rp,Rs);
[bbp,abp]=cheby2(n,Rs,Wn);
clear pass* stop* Rp Rs W* n;

%% Detection with Density Calculation
clear datapoints swa_results channels datax signal dataff EEGder EEG difference ;
clear pos_index neg_index troughs peaks poscross negcross wndx b bx c cx nump maxb maxc lastpk;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp([' Analyzing ', num2str(size(LoadedEEG.data,1)),' channels']);
fprintf('\n');

% 预计算总时间
total_time_global = size(datainput, 2) / orig_fs;

for i=1:size(LoadedEEG.data,1)
    fprintf('... channel %3.0f/%3.0f - %3.0f%%',i,size(LoadedEEG.data,1),0)
    
    % 原有数据处理
    datax = squeeze(LoadedEEG.data(i,1:size(LoadedEEG.data,2),1));
	signal = resample(double(datax),fs,orig_fs);
    EEG=filtfilt(bbp, abp, signal);
    datapoints=length(EEG);
    channels(i).datalength=datapoints;
    
    %% Search Negcross and Poscross
    pos_index=zeros(length(EEG),1);
    pos_index(find(EEG>0))=1;
    difference=diff(pos_index);
    poscross=find(difference==1);
    negcross=find(difference==-1);
    EEGder=meanfilt(diff(EEG),5);
    pos_index=zeros(length(EEGder),1);
    pos_index(find(EEGder>0))=1;
    difference=diff(pos_index);
    peaks=find(difference==-1)+1;
    troughs=find(difference==1)+1;
    peaks(EEG(peaks)<0)=[];
    troughs(EEG(troughs)>0)=[];
    
    %% Makes negcross and poscross same size to start
    if isempty(negcross) || isempty(poscross)
        continue;
    end
    if negcross(1)<poscross(1); 
            start=1;
        else
            start=2;
    end; 
    
    if start==2;
            poscross(1)=[];
    end;
        lastpk=NaN;
        ch=i;
                
%% Wave parameters initialization
	channels(ch).negzx{1}=[];
	channels(ch).poszx{1}=[];
	channels(ch).wvend{1}=[];
	channels(ch).negpks{1}=[];
	channels(ch).maxnegpk{1}=[];
	channels(ch).negpkamp{1}=[];
	channels(ch).maxnegpkamp{1}=[];
	channels(ch).pospks{1}=[];
	channels(ch).maxpospk{1}=[];
	channels(ch).pospkamp{1}=[];
	channels(ch).maxpospkamp{1}=[];
	channels(ch).mxdnslp{1}=[];
	channels(ch).mxupslp{1}=[];
	channels(ch).maxampwn{1}=[];
	channels(ch).minampwn{1}=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% Locate Peaks
	wvi=1;
	for wndx=start:length(negcross)-1
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b... channel %3.0f/%3.0f - %3.0f%%',i,size(LoadedEEG.data,1),round(100*wndx/(length(negcross)-1)))
    wavest=negcross(wndx);
    wavend=negcross(wndx+1);
        
    mxdn=abs(min(meanfilt(diff(EEG(wavest:poscross(wndx))),5)))*fs;
    mxup=max(meanfilt(diff(EEG(wavest:poscross(wndx))),5))*fs;
    negpeaks=troughs(troughs>wavest&troughs<wavend);
    wavepk=negpeaks(EEG(negpeaks)==min(EEG(negpeaks)));
    
    pospeaks=peaks(peaks>wavest&peaks<=wavend);

    if isempty(pospeaks)
        pospeaks=wavend; 
    end;
                        
    poszx=poscross(wndx);
    b=EEG(negpeaks);  
    bx=negpeaks;
    c=EEG(pospeaks);
    cx=pospeaks;
    nump=length(negpeaks);

    maxampwn=max(EEG(max(wavest-128,1):min(poszx+128,length(EEG))));
    minampwn=min(EEG(max(wavest-128,1):min(poszx+128,length(EEG))));
    maxb=min(EEG(negpeaks));
    if maxb>0
        maxb=maxb(1);
    end;            
    maxbx=negpeaks(EEG(negpeaks)==maxb);

    maxc=max(EEG(pospeaks));             
    if maxc>0
        maxc=maxc(1);
    end;            
    maxcx=pospeaks(EEG(pospeaks)==maxc);

    lastpk=maxcx;

	waveamp=abs(single(maxc))+abs(single(maxb));
    wavelength=abs((single(wavest)-single(poszx))./fs);
	if wavelength>0.1 && wavelength<1.0
        if waveamp>thramp
    
%% Wave parameters
    channels(ch).negzx{wvi}=round((single(wavest)./128).*orig_fs);
    channels(ch).poszx{wvi}=round((single(poszx)./128).*orig_fs);
    channels(ch).wvend{wvi}=round((single(wavend)./128).*orig_fs);
    channels(ch).negpks{wvi}={round((single(bx)./128).*orig_fs)};
    channels(ch).maxnegpk{wvi}=round((single(maxbx)./128).*orig_fs);
    channels(ch).negpkamp{wvi}=single(b);
    channels(ch).maxnegpkamp{wvi}=single(maxb);
    channels(ch).pospks{wvi}={round((single(cx)./128).*orig_fs)};
    channels(ch).maxpospk{wvi}=round((single(maxcx)./128).*orig_fs);
    channels(ch).pospkamp{wvi}=single(c);
    channels(ch).maxpospkamp{wvi}=single(maxc);
    channels(ch).mxdnslp{wvi}=single(mxdn);
    channels(ch).mxupslp{wvi}=single(mxup);
    channels(ch).maxampwn{wvi}=single(maxampwn);
    channels(ch).minampwn{wvi}=single(minampwn);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

     wvi=wvi+1;

        end;
    end;
    
    clear wavest wavend poszx bx maxbx b maxb cx maxcx x maxc mxdn mxup nump negpeaks pospeaks;
    clear wavelength waveamp;
    
     end;
     
    % ====== 新增：通道级密度计算 ======
    num_waves = wvi - 1;  % 该通道的慢波数量
    if total_time_global > 0
        sw_density = num_waves / total_time_global; % 波/秒
    else
        sw_density = 0;
    end
    
    % 保存通道级密度指标
    channels(i).sw_count = num_waves;
    channels(i).sw_density = sw_density;
    channels(i).mean_wavelength = total_time_global / max(1, num_waves);
    
    % 更新全局计数
    twa_results.global_sw_count = twa_results.global_sw_count + num_waves;
    
    fprintf('\n');
end;

%% 新增：全局密度计算
num_channels = size(LoadedEEG.data,1);
if num_channels > 0 && total_time_global > 0
    % 平均慢波密度(波/秒/通道)
    twa_results.global_sw_density = twa_results.global_sw_count / (total_time_global * num_channels);
    
    % 平均慢波持续时间
    twa_results.mean_duration = total_time_global / max(1, twa_results.global_sw_count);
else
    twa_results.global_sw_density = 0;
    twa_results.mean_duration = 0;
end

%% Save Output
twa_results.channels = channels;
twa_results.recording_duration = total_time_global;
twa_results.sampling_rate = orig_fs;

disp(['**** Detection Completed - Total Waves: ', num2str(twa_results.global_sw_count)]);
disp(['**** Average SW Density: ', num2str(twa_results.global_sw_density), ' waves/sec/channel']);
end
