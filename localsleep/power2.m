group1_dir1 = 'D:\下载\T2ant\T2ant';    
group1_files = dir([group1_dir1, filesep, '*.set']);  

theta_range = [4 8];
beta_range = [14 30];
delta_range = [1 4];

Fs = 500;
n_subjects = length(group1_files);

% 初始化结果结构
absolute_power = struct('theta', [], 'beta', [], 'delta', []);
relative_power = struct('theta', [], 'beta', [], 'delta', []);
theta_beta_ratio = struct('absolute_theta_beta_ratio', [], 'relative_theta_beta_ratio', []);

for i=1:n_subjects
    
    subj_fn = group1_files(i).name;
    sub = subj_fn(1:end-4);
    EEG = pop_loadset('filename',strcat(sub, '.set'), 'filepath', group1_dir1); %导入数据
    
    eeg_data = EEG.data;
    [n_channels, n_timepoints, n_epochs] = size(eeg_data);
    
     % 初始化每个被试的结果
    absolute_power_theta = zeros(n_channels, 1);
    absolute_power_beta = zeros(n_channels, 1);
    absolute_power_delta = zeros(n_channels, 1);
    
    relative_power_theta = zeros(n_channels, 1);
    relative_power_beta = zeros(n_channels, 1);
    relative_power_delta = zeros(n_channels, 1);
    
    absolute_theta_beta_ratio_values = zeros(n_channels, 1);
    relative_theta_beta_ratio_values = zeros(n_channels, 1);
    
    for ch = 1:n_channels
        
        % 初始化累加器
        total_absolute_power = struct('theta', 0, 'beta', 0, 'delta', 0);
        total_relative_power = struct('theta', 0, 'beta', 0, 'delta', 0);
        total_absolute_theta_beta_ratio = 0;
        total_relative_theta_beta_ratio = 0;
        
        for epoch = 1:n_epochs
            
            data = eeg_data(ch, :, epoch);
            
            [psd, freqs] = pwelch(data, [], [], [], Fs);
            
            absolute_theta_power = bandpower(psd, freqs, theta_range, 'psd');
            absolute_beta_power = bandpower(psd, freqs, beta_range, 'psd');
            absolute_delta_power = bandpower(psd, freqs, delta_range, 'psd');
            absolute_theta_beta_ratio_power = absolute_theta_power / absolute_beta_power;
            
            total_power = bandpower(psd, freqs, [1 47], 'psd');
            
            relative_theta_power = absolute_theta_power / total_power;
            relative_beta_power = absolute_beta_power / total_power;
            relative_delta_power = absolute_delta_power / total_power;
            relative_theta_beta_ratio_power = relative_theta_power / relative_beta_power;
            
            % 累加绝对功率
            total_absolute_power.theta = total_absolute_power.theta + absolute_theta_power;
            total_absolute_power.beta = total_absolute_power.beta + absolute_beta_power;
            total_absolute_power.delta = total_absolute_power.delta + absolute_delta_power;
            
            % 累加相对功率
            total_relative_power.theta = total_relative_power.theta + relative_theta_power;
            total_relative_power.beta = total_relative_power.beta + relative_beta_power;
            total_relative_power.delta = total_relative_power.delta + relative_delta_power;
            
            % 累加θ/β比值
            total_absolute_theta_beta_ratio = total_absolute_theta_beta_ratio + absolute_theta_beta_ratio_power;
            total_relative_theta_beta_ratio = total_relative_theta_beta_ratio + relative_theta_beta_ratio_power;
            
        end
        
        % 计算平均值
        absolute_power_theta(ch) = total_absolute_power.theta / n_epochs;
        absolute_power_beta(ch) = total_absolute_power.beta / n_epochs;
        absolute_power_delta(ch) = total_absolute_power.delta / n_epochs;

        relative_power_theta(ch) = total_relative_power.theta / n_epochs;
        relative_power_beta(ch) = total_relative_power.beta / n_epochs;
        relative_power_delta(ch) = total_relative_power.delta / n_epochs;

        absolute_theta_beta_ratio_values(ch) = total_absolute_theta_beta_ratio / n_epochs;
        relative_theta_beta_ratio_values(ch) = total_relative_theta_beta_ratio / n_epochs;
    end
    
    % 保存每个被试的结果
    absolute_power(i).theta = absolute_power_theta;
    absolute_power(i).beta = absolute_power_beta;
    absolute_power(i).delta = absolute_power_delta;

    relative_power(i).theta = relative_power_theta;
    relative_power(i).beta = relative_power_beta;
    relative_power(i).delta = relative_power_delta;

    theta_beta_ratio(i).absolute_theta_beta_ratio = absolute_theta_beta_ratio_values;
    theta_beta_ratio(i).relative_theta_beta_ratio = relative_theta_beta_ratio_values;
    
end

% 保存到文件
save('mean_power_results_all_subjects.mat', 'absolute_power', 'relative_power', 'theta_beta_ratio');