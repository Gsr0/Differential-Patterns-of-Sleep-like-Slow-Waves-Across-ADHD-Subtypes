% 定义脑区电极编号
frontalElectrodes = [1, 2, 3, 5, 6, 7];  % 额叶电极
parietalElectrodes = [9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21]; % 顶叶电极
occipitalElectrodes = [23, 24, 25, 27, 28, 29, 30];    % 枕叶电极
temporalElectrodes = [4, 8, 13, 17, 22, 26];           % 颞叶电极

% 加载 .mat 文件
load('mean_power_results_all_subjects(3).mat');

% 获取被试数量
n_subjects = length(absolute_power);

% 获取通道数量
n_channels = length(absolute_power(1).theta);

% 准备 Excel 数据表
absolute_power_data = cell(n_subjects * n_channels, 5); % 5列用于存储绝对功率数据
relative_power_data = cell(n_subjects * n_channels, 5); % 5列用于存储相对功率数据
theta_beta_ratio_data = cell(n_subjects * n_channels, 6); % 4列用于存储θ/β比值数据

% 填充绝对功率数据
row = 1;
for i = 1:n_subjects
    for ch = 1:n_channels
        absolute_power_data{row, 1} = i; % 被试编号
        absolute_power_data{row, 2} = ch; % 通道编号
        absolute_power_data{row, 3} = absolute_power(i).theta(ch);
        absolute_power_data{row, 4} = absolute_power(i).beta(ch);
        absolute_power_data{row, 5} = absolute_power(i).delta(ch);
        row = row + 1;
    end
end

% 填充相对功率数据
row = 1;
for i = 1:n_subjects
    for ch = 1:n_channels
        relative_power_data{row, 1} = i; % 被试编号
        relative_power_data{row, 2} = ch; % 通道编号
        relative_power_data{row, 3} = relative_power(i).theta(ch);
        relative_power_data{row, 4} = relative_power(i).beta(ch);
        relative_power_data{row, 5} = relative_power(i).delta(ch);
        row = row + 1;
    end
end

% 填充θ/β比值数据
row = 1;
for i = 1:n_subjects
    for ch = 1:n_channels
        theta_beta_ratio_data{row, 1} = i; % 被试编号
        theta_beta_ratio_data{row, 2} = ch; % 通道编号
        theta_beta_ratio_data{row, 3} = theta_beta_ratio(i).absolute_theta_beta_ratio(ch);
        theta_beta_ratio_data{row, 4} = theta_beta_ratio(i).relative_theta_beta_ratio(ch);
    end
    row = row + 1;
end

% 计算每个被试所有通道的平均值
absolute_power_avg = zeros(n_subjects, 3); % 3列：theta, beta, delta
relative_power_avg = zeros(n_subjects, 3);
theta_beta_ratio_avg = zeros(n_subjects, 4); % 2列：absolute, relative

for i = 1:n_subjects
    absolute_power_avg(i, 1) = mean(absolute_power(i).theta);
    absolute_power_avg(i, 2) = mean(absolute_power(i).beta);
    absolute_power_avg(i, 3) = mean(absolute_power(i).delta);

    relative_power_avg(i, 1) = mean(relative_power(i).theta);
    relative_power_avg(i, 2) = mean(relative_power(i).beta);
    relative_power_avg(i, 3) = mean(relative_power(i).delta);

    frontalElectrodes_num = 0;
    parietalElectrodes_num = 0;
    occipitalElectrodes_num = 0;
    temporalElectrodes_num = 0;

    for ch = 1:n_channels
        if ismember(ch, frontalElectrodes)
        frontalElectrodes_num = frontalElectrodes_num + theta_beta_ratio(i).absolute_theta_beta_ratio(ch);
        end
        if ismember(ch, parietalElectrodes)
        parietalElectrodes_num = parietalElectrodes_num + theta_beta_ratio(i).absolute_theta_beta_ratio(ch);
        end
        if ismember(ch, occipitalElectrodes)
        occipitalElectrodes_num = occipitalElectrodes_num + theta_beta_ratio(i).absolute_theta_beta_ratio(ch);
        end
        if ismember(ch, temporalElectrodes)
        temporalElectrodes_num = temporalElectrodes_num + theta_beta_ratio(i).absolute_theta_beta_ratio(ch);
        end
    end
    theta_beta_ratio_avg(i, 1) = frontalElectrodes_num / 6;
    theta_beta_ratio_avg(i, 2) = parietalElectrodes_num / 11;
    theta_beta_ratio_avg(i, 3) = occipitalElectrodes_num / 7;
    theta_beta_ratio_avg(i, 4) = temporalElectrodes_num / 6;

end

% 创建包含被试编号的平均值表格
absolute_power_avg_table = [(1:n_subjects)', absolute_power_avg];
relative_power_avg_table = [(1:n_subjects)', relative_power_avg];
theta_beta_ratio_avg_table = [(1:n_subjects)', theta_beta_ratio_avg];

% 保存平均值到 Excel 文件
xlswrite('absolute_power_avg_results.xlsx', [{'Subject', 'Average Theta Power', 'Average Beta Power', 'Average Delta Power'}; num2cell(absolute_power_avg_table)]);
xlswrite('relative_power_avg_results.xlsx', [{'Subject', 'Average Relative Theta Power', 'Average Relative Beta Power', 'Average Relative Delta Power'}; num2cell(relative_power_avg_table)]);
xlswrite('theta_beta_ratio_avg_results1.xlsx', [{'Subject', '额叶', '顶叶', '枕叶', '颞叶'}; num2cell(theta_beta_ratio_avg_table)]);

disp('Data saved to absolute_power_results.xlsx, relative_power_results.xlsx, and theta_beta_ratio_results1.xlsx');