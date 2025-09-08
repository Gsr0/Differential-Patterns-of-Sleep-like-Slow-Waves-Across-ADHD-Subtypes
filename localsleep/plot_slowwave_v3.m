batch_process_slowwaves('D:\桌面\T0-ANT', 'D:\桌面\T0-ANT');
%
% % 自定义参数
% batch_process_slowwaves('C:\EEG_Data\Input', 'C:\EEG_Data\Output', ...
%     'freq_range', [0.5 4], ...
%     'amplitude_threshold', 80, ...
%     'duration_range', [0.5 2], ...
%     'channels', [1:32], ...
%     'detection_method', 'amplitude', ...
%     'time_window', [-0.5 1], ...
%     'separate_channels', false, ...
%     'file_pattern', '*.set', ...
%     'min_events', 5);
%
% % 分通道绘图
% batch_process_slowwaves('C:\EEG_Data\Input', 'C:\EEG_Data\Output', ...
%     'separate_channels', true, ...
%     'channels', [1:8]);


function batch_process_slowwaves(input_folder, output_folder, varargin)
% 批量处理文件夹中的所有EEG文件，检测慢波并生成平均波形图
% 
% 输入参数:
%   input_folder: 输入文件夹路径（包含.set文件）
%   output_folder: 输出文件夹路径
%   可选参数:
%     'freq_range': 频率范围 [低频 高频] (默认: [0.5 4])
%     'amplitude_threshold': 振幅阈值,单位uV (默认: 75)
%     'duration_range': 持续时间范围,单位秒 [最短 最长] (默认: [0.5 2])
%     'channels': 要分析的通道 (默认: 所有通道)
%     'detection_method': 检测方法 'amplitude'或'wavelet' (默认: 'amplitude')
%     'time_window': 绘图时间窗口 (默认: [-0.5 1])
%     'separate_channels': 是否分通道绘图 (默认: false)
%     'file_pattern': 文件匹配模式 (默认: '*.set')
%     'min_events': 绘图所需最少事件数 (默认: 5)

% 解析输入参数
p = inputParser;
addParameter(p, 'freq_range', [0.5 4], @(x) length(x)==2 && x(1)<x(2));
addParameter(p, 'amplitude_threshold', 75, @(x) x>0);
addParameter(p, 'duration_range', [0.5 2], @(x) length(x)==2 && x(1)<x(2));
addParameter(p, 'channels', [], @(x) isnumeric(x) || isempty(x));
addParameter(p, 'detection_method', 'amplitude', @(x) ismember(x, {'amplitude', 'wavelet'}));
addParameter(p, 'time_window', [-0.5 1], @(x) length(x)==2 && x(1)<x(2));
addParameter(p, 'separate_channels', false, @islogical);
addParameter(p, 'file_pattern', '*.set', @ischar);
addParameter(p, 'min_events', 5, @(x) x>0);
parse(p, varargin{:});

% 获取参数
freq_range = p.Results.freq_range;
amp_threshold = p.Results.amplitude_threshold;
duration_range = p.Results.duration_range;
channels = p.Results.channels;
method = p.Results.detection_method;
time_window = p.Results.time_window;
separate_channels = p.Results.separate_channels;
file_pattern = p.Results.file_pattern;
min_events = p.Results.min_events;

try
    % 检查输入文件夹
    if ~exist(input_folder, 'dir')
        error('输入文件夹不存在: %s', input_folder);
    end
    
    % 创建输出文件夹
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
        fprintf('创建输出文件夹: %s\n', output_folder);
    end
    
    % 创建子文件夹
    marked_folder = fullfile(output_folder, 'marked_data');
    plots_folder = fullfile(output_folder, 'plots');
    if ~exist(marked_folder, 'dir'), mkdir(marked_folder); end
    if ~exist(plots_folder, 'dir'), mkdir(plots_folder); end
    
    % 查找所有.set文件
    file_list = dir(fullfile(input_folder, file_pattern));
    if isempty(file_list)
        error('在文件夹 %s 中未找到 %s 文件', input_folder, file_pattern);
    end
    
    fprintf('找到 %d 个文件需要处理\n', length(file_list));
    
    % 初始化结果统计
    results_summary = struct();
    results_summary.total_files = length(file_list);
    results_summary.successful_detections = 0;
    results_summary.successful_plots = 0;
    results_summary.failed_files = {};
    results_summary.detection_stats = [];
    
    % 处理每个文件
    for i = 1:length(file_list)
        filename = file_list(i).name;
        [~, basename, ~] = fileparts(filename);
        input_file = fullfile(input_folder, filename);
        
        fprintf('\n=== 处理文件 %d/%d: %s ===\n', i, length(file_list), filename);
        
        try
            % 步骤1: 检测慢波
            marked_file = fullfile(marked_folder, [basename '_marked.set']);
            fprintf('正在检测慢波...\n');
            
            [filtered_set, detection_stats] = detect_slowwaves_single_file(input_file, marked_file, ...
                freq_range, amp_threshold, duration_range, channels, method);
            
            if ~isempty(detection_stats) && detection_stats.num_events >= min_events
                results_summary.successful_detections = results_summary.successful_detections + 1;
                results_summary.detection_stats = [results_summary.detection_stats, detection_stats];
                
                % 步骤2: 绘制平均波形
                fprintf('正在生成平均波形图...\n');
                if separate_channels
                    plot_file = fullfile(plots_folder, [basename '_slowwaves_channels.png']);
                else
                    plot_file = fullfile(plots_folder, [basename '_slowwave_average.png']);
                end
                
                success = plot_slowwave_average(marked_file, plot_file, time_window, ...
                    channels, separate_channels, min_events);
                
                if success
                    results_summary.successful_plots = results_summary.successful_plots + 1;
                    fprintf('✓ 成功生成图片: %s\n', [basename '_slowwave*.png']);
                else
                    fprintf('✗ 图片生成失败\n');
                end
            else
                fprintf('✗ 检测到的慢波数量不足 (需要至少%d个，实际%d个)\n', ...
                    min_events, detection_stats.num_events);
                results_summary.failed_files{end+1} = sprintf('%s (慢波不足)', filename);
            end
            
        catch ME
            fprintf('✗ 处理文件失败: %s\n', ME.message);
            results_summary.failed_files{end+1} = sprintf('%s (错误: %s)', filename, ME.message);
        end
    end
    
    % 生成总结报告
    generate_summary_report(results_summary, output_folder);
    
    fprintf('\n=== 批量处理完成 ===\n');
    fprintf('总文件数: %d\n', results_summary.total_files);
    fprintf('成功检测: %d\n', results_summary.successful_detections);
    fprintf('成功绘图: %d\n', results_summary.successful_plots);
    fprintf('失败文件: %d\n', length(results_summary.failed_files));
    fprintf('结果保存在: %s\n', output_folder);
    
catch ME
    fprintf('批量处理出错: %s\n', ME.message);
    rethrow(ME);
end

end

function [filtered_EEG, stats] = detect_slowwaves_single_file(input_file, output_file, ...
    freq_range, amp_threshold, duration_range, channels, method)
% 单个文件的慢波检测
    
    % 加载EEG数据
    EEG = pop_loadset(input_file);
    
    if isempty(EEG.data)
        error('EEG数据为空');
    end
    
    % 设置通道
    if isempty(channels)
        channels = 1:EEG.nbchan;
    else
        channels = channels(channels <= EEG.nbchan);
    end
    
    % 预处理
    EEG_filtered = preprocess_eeg(EEG, freq_range);
    
    % 检测慢波
    switch method
        case 'amplitude'
            slowwave_events = detect_slowwaves_amplitude(EEG_filtered, channels, ...
                amp_threshold, duration_range);
        case 'wavelet'
            slowwave_events = detect_slowwaves_wavelet(EEG_filtered, channels, ...
                freq_range, amp_threshold, duration_range);
    end
    
    % 创建统计信息
    stats = struct();
    stats.filename = input_file;
    stats.num_events = length(slowwave_events);
    stats.channels_involved = length(unique([slowwave_events.channel]));
    
    if ~isempty(slowwave_events)
        durations = [slowwave_events.duration];
        stats.mean_duration = mean(durations);
        stats.std_duration = std(durations);
        
        amplitudes = [];
        for i = 1:length(slowwave_events)
            if isfield(slowwave_events(i), 'peak_amplitude')
                amplitudes = [amplitudes, slowwave_events(i).peak_amplitude];
            end
        end
        
        if ~isempty(amplitudes)
            stats.mean_amplitude = mean(amplitudes);
            stats.std_amplitude = std(amplitudes);
        else
            stats.mean_amplitude = NaN;
            stats.std_amplitude = NaN;
        end
    else
        stats.mean_duration = NaN;
        stats.std_duration = NaN;
        stats.mean_amplitude = NaN;
        stats.std_amplitude = NaN;
    end
    
    % 创建标记的EEG数据并保存
    filtered_EEG = create_marked_eeg(EEG, slowwave_events);
    pop_saveset(filtered_EEG, output_file);
end

function success = plot_slowwave_average(marked_file, output_file, time_window, ...
    channels, separate_channels, min_events)
% 绘制单个文件的平均慢波
    success = false;
    
    try
        % 加载标记的数据
        EEG = pop_loadset(marked_file);
        
        % 查找慢波事件
        slowwave_events = [];
        if ~isempty(EEG.event)
            event_types = {EEG.event.type};
            slowwave_idx = strcmp(event_types, 'slowwave');
            if any(slowwave_idx)
                slowwave_events = EEG.event(slowwave_idx);
            end
        end
        
        if length(slowwave_events) < min_events
            return;
        end
        
        % 设置通道
        if isempty(channels)
            if isfield(slowwave_events, 'channel')
                channels = unique([slowwave_events.channel]);
            else
                channels = 1:min(32, EEG.nbchan);
            end
        end
        
        % 绘制图形
        if separate_channels
            plot_by_channels(EEG, slowwave_events, channels, time_window, output_file, min_events);
        else
            plot_averaged_all(EEG, slowwave_events, channels, time_window, output_file, min_events);
        end
        
        success = true;
        
    catch ME
        fprintf('绘图失败: %s\n', ME.message);
    end
end

function plot_averaged_all(EEG, events, channels, time_window, save_path, min_events)
% 绘制平均慢波（所有通道合并）
    srate = EEG.srate;
    window_samples = round(time_window * srate);
    time_vec = (window_samples(1):window_samples(2)) / srate;
    
    % 提取所有慢波片段
    all_segments = [];
    valid_events = 0;
    
    for i = 1:length(events)
        if isfield(events(i), 'channel')
            ch = events(i).channel;
        else
            ch = channels(1);
        end
        
        if ismember(ch, channels)
            if isfield(events(i), 'peak_sample')
                peak_sample = events(i).peak_sample;
            else
                peak_sample = events(i).latency;
            end
            
            start_sample = peak_sample + window_samples(1);
            end_sample = peak_sample + window_samples(2);
            
            if start_sample > 0 && end_sample <= size(EEG.data, 2)
                segment = EEG.data(ch, start_sample:end_sample);
                all_segments = [all_segments; segment];
                valid_events = valid_events + 1;
            end
        end
    end
    
    if valid_events < min_events
        return;
    end
    
    % 计算平均和标准误
    mean_wave = mean(all_segments, 1);
    std_wave = std(all_segments, 1);
    sem_wave = std_wave / sqrt(size(all_segments, 1));
    
    % 创建图形
    figure('Position', [100, 100, 1200, 800], 'Color', 'white');
    
    hold on;
    
    % 设置颜色
    color_main = [0.2 0.4 0.8];  % 蓝色
    
    % 绘制标准误阴影
    fill([time_vec, fliplr(time_vec)], ...
         [mean_wave + sem_wave, fliplr(mean_wave - sem_wave)], ...
         color_main, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    
    % 绘制平均线
    plot(time_vec, mean_wave, 'Color', color_main, 'LineWidth', 3);
    
    % 添加参考线
    plot([time_window(1), time_window(2)], [0, 0], 'k--', 'LineWidth', 1);
    plot([0, 0], [min(mean_wave-sem_wave)*1.2, max(mean_wave+sem_wave)*1.2], 'k--', 'LineWidth', 1);
    
    % 设置坐标轴
    xlim(time_window);
    ylim([min(mean_wave-sem_wave)*1.2, max(mean_wave+sem_wave)*1.2]);
    
    % 美化图形
    xlabel('Time (s)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Voltage (μV)', 'FontSize', 14, 'FontWeight', 'bold');
    title(sprintf('Averaged Slow-Wave\n(n=%d events)', valid_events), ...
          'FontSize', 16, 'FontWeight', 'bold', 'Color', color_main);
    
    grid on;
    set(gca, 'GridAlpha', 0.3);
    set(gca, 'FontSize', 12);
    set(gca, 'LineWidth', 1.5);
    
    % 添加统计信息
    [~, filename, ~] = fileparts(save_path);
    text(0.02, 0.98, sprintf('File: %s\nPeak: %.1f μV\nEvents: %d', ...
        strrep(filename, '_', '\_'), min(mean_wave), valid_events), ...
        'Units', 'normalized', 'VerticalAlignment', 'top', ...
        'FontSize', 10, 'BackgroundColor', 'white', 'EdgeColor', 'black');
    
    hold off;
    
    % 保存图片
    saveas(gcf, save_path, 'png');
    saveas(gcf, strrep(save_path, '.png', '.fig'));
    close(gcf);
end

function plot_by_channels(EEG, events, channels, time_window, save_path, min_events)
% 分通道绘制平均慢波
    srate = EEG.srate;
    window_samples = round(time_window * srate);
    time_vec = (window_samples(1):window_samples(2)) / srate;
    
    % 计算子图布局
    n_channels = length(channels);
    n_cols = ceil(sqrt(n_channels));
    n_rows = ceil(n_channels / n_cols);
    
    % 创建图形
    figure('Position', [100, 100, 1200, 800], 'Color', 'white');
    
    color_main = [0.2 0.4 0.8];
    
    for ch_idx = 1:length(channels)
        ch = channels(ch_idx);
        
        % 提取该通道的慢波片段
        ch_segments = [];
        for i = 1:length(events)
            if isfield(events(i), 'channel') && events(i).channel == ch
                if isfield(events(i), 'peak_sample')
                    peak_sample = events(i).peak_sample;
                else
                    peak_sample = events(i).latency;
                end
                
                start_sample = peak_sample + window_samples(1);
                end_sample = peak_sample + window_samples(2);
                
                if start_sample > 0 && end_sample <= size(EEG.data, 2)
                    segment = EEG.data(ch, start_sample:end_sample);
                    ch_segments = [ch_segments; segment];
                end
            end
        end
        
        % 创建子图
        subplot(n_rows, n_cols, ch_idx);
        
        if size(ch_segments, 1) >= min_events
            % 计算平均和标准误
            mean_wave = mean(ch_segments, 1);
            std_wave = std(ch_segments, 1);
            sem_wave = std_wave / sqrt(size(ch_segments, 1));
            
            hold on;
            
            % 绘制标准误阴影
            fill([time_vec, fliplr(time_vec)], ...
                 [mean_wave + sem_wave, fliplr(mean_wave - sem_wave)], ...
                 color_main, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
            
            % 绘制平均线
            plot(time_vec, mean_wave, 'Color', color_main, 'LineWidth', 2);
            
            % 添加参考线
            plot([time_window(1), time_window(2)], [0, 0], 'k--', 'LineWidth', 0.8);
            plot([0, 0], [min(mean_wave-sem_wave)*1.1, max(mean_wave+sem_wave)*1.1], 'k--', 'LineWidth', 0.8);
            
            xlim(time_window);
            ylim([min(mean_wave-sem_wave)*1.1, max(mean_wave+sem_wave)*1.1]);
            
            title(sprintf('Ch%d (n=%d)', ch, size(ch_segments, 1)), 'FontSize', 10);
            
            hold off;
        else
            text(0.5, 0.5, sprintf('Ch%d\nInsufficient\n(n=%d)', ch, size(ch_segments, 1)), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                'Units', 'normalized', 'FontSize', 9);
            xlim(time_window);
            ylim([-1, 1]);
        end
        
        % 设置坐标轴标签
        if ch_idx > n_channels - n_cols
            xlabel('Time (s)', 'FontSize', 10);
        end
        if mod(ch_idx-1, n_cols) == 0
            ylabel('Voltage (μV)', 'FontSize', 10);
        end
        
        grid on;
        set(gca, 'FontSize', 9);
    end
    
    [~, filename, ~] = fileparts(save_path);
    sgtitle(sprintf('Slow-Waves by Channel - %s', strrep(filename, '_', '\_')), ...
        'FontSize', 14, 'FontWeight', 'bold');
    
    % 保存图片
    saveas(gcf, save_path, 'png');
    saveas(gcf, strrep(save_path, '.png', '.fig'));
    close(gcf);
end

function generate_summary_report(results, output_folder)
% 生成总结报告
    report_file = fullfile(output_folder, 'processing_summary.txt');
    
    fid = fopen(report_file, 'w');
    if fid == -1
        fprintf('无法创建报告文件\n');
        return;
    end
    
    fprintf(fid, '=== EEG慢波批量处理报告 ===\n');
    fprintf(fid, '处理时间: %s\n\n', datestr(now));
    
    fprintf(fid, '总体统计:\n');
    fprintf(fid, '  总文件数: %d\n', results.total_files);
    fprintf(fid, '  成功检测: %d\n', results.successful_detections);
    fprintf(fid, '  成功绘图: %d\n', results.successful_plots);
    fprintf(fid, '  失败文件: %d\n\n', length(results.failed_files));
    
    if ~isempty(results.detection_stats)
        events = [results.detection_stats.num_events];
        durations = [results.detection_stats.mean_duration];
        amplitudes = [results.detection_stats.mean_amplitude];
        
        fprintf(fid, '检测统计 (成功文件):\n');
        fprintf(fid, '  平均事件数: %.1f ± %.1f\n', mean(events), std(events));
        fprintf(fid, '  平均持续时间: %.2f ± %.2f 秒\n', nanmean(durations), nanstd(durations));
        fprintf(fid, '  平均振幅: %.1f ± %.1f μV\n\n', nanmean(amplitudes), nanstd(amplitudes));
    end
    
    if ~isempty(results.failed_files)
        fprintf(fid, '失败文件列表:\n');
        for i = 1:length(results.failed_files)
            fprintf(fid, '  %s\n', results.failed_files{i});
        end
    end
    
    fclose(fid);
    fprintf('处理报告已保存: %s\n', report_file);
end

% 包含之前的辅助函数
function EEG_filtered = preprocess_eeg(EEG, freq_range)
    EEG_filtered = pop_eegfiltnew(EEG, freq_range(1), freq_range(2));
end

function events = detect_slowwaves_amplitude(EEG, channels, threshold, duration_range)
    events = [];
    srate = EEG.srate;
    min_samples = round(duration_range(1) * srate);
    
    for ch = channels
        data = EEG.data(ch, :);
        
        [neg_peaks, neg_locs] = findpeaks(-data, 'MinPeakHeight', threshold, ...
            'MinPeakDistance', min_samples);
        
        for i = 1:length(neg_locs)
            peak_loc = neg_locs(i);
            peak_val = -neg_peaks(i);
            
            [start_idx, end_idx] = find_wave_boundaries(data, peak_loc, threshold/2);
            
            duration = (end_idx - start_idx) / srate;
            
            if duration >= duration_range(1) && duration <= duration_range(2)
                event = struct();
                event.channel = ch;
                event.start_sample = start_idx;
                event.end_sample = end_idx;
                event.peak_sample = peak_loc;
                event.peak_amplitude = peak_val;
                event.duration = duration;
                event.start_time = (start_idx-1) / srate;
                event.end_time = (end_idx-1) / srate;
                
                events = [events, event];
            end
        end
    end
end

function events = detect_slowwaves_wavelet(EEG, channels, freq_range, threshold, duration_range)
    events = [];
    srate = EEG.srate;
    frequencies = freq_range(1):0.1:freq_range(2);
    
    for ch = channels
        data = EEG.data(ch, :);
        [wt, ~] = cwt(data, frequencies, srate);
        power = abs(wt).^2;
        mean_power = mean(power, 1);
        
        [peaks, locs] = findpeaks(mean_power, 'MinPeakHeight', threshold^2);
        
        for i = 1:length(locs)
            peak_loc = locs(i);
            [start_idx, end_idx] = find_wave_boundaries_power(mean_power, peak_loc, peaks(i)/2);
            
            duration = (end_idx - start_idx) / srate;
            
            if duration >= duration_range(1) && duration <= duration_range(2)
                event = struct();
                event.channel = ch;
                event.start_sample = start_idx;
                event.end_sample = end_idx;
                event.peak_sample = peak_loc;
                event.peak_power = peaks(i);
                event.duration = duration;
                event.start_time = (start_idx-1) / srate;
                event.end_time = (end_idx-1) / srate;
                
                events = [events, event];
            end
        end
    end
end

function [start_idx, end_idx] = find_wave_boundaries(data, peak_loc, threshold)
    start_idx = peak_loc;
    while start_idx > 1 && abs(data(start_idx)) > threshold
        start_idx = start_idx - 1;
    end
    
    end_idx = peak_loc;
    while end_idx < length(data) && abs(data(end_idx)) > threshold
        end_idx = end_idx + 1;
    end
end

function [start_idx, end_idx] = find_wave_boundaries_power(power_data, peak_loc, threshold)
    start_idx = peak_loc;
    while start_idx > 1 && power_data(start_idx) > threshold
        start_idx = start_idx - 1;
    end
    
    end_idx = peak_loc;
    while end_idx < length(power_data) && power_data(end_idx) > threshold
        end_idx = end_idx + 1;
    end
end

function marked_EEG = create_marked_eeg(EEG, events)
    marked_EEG = EEG;
    
    if isempty(marked_EEG.event)
        marked_EEG.event = struct('type', {}, 'latency', {}, 'duration', {});
    end
    
    for i = 1:length(events)
        event_idx = length(marked_EEG.event) + 1;
        marked_EEG.event(event_idx).type = 'slowwave';
        marked_EEG.event(event_idx).latency = events(i).start_sample;
        marked_EEG.event(event_idx).duration = events(i).end_sample - events(i).start_sample;
        marked_EEG.event(event_idx).channel = events(i).channel;
        if isfield(events(i), 'peak_amplitude')
            marked_EEG.event(event_idx).peak_amplitude = events(i).peak_amplitude;
        end
    end
    
    marked_EEG.urevent = marked_EEG.event;
end

% ====== 使用示例 ======
%
% % 基本使用 - 处理整个文件夹
% batch_process_slowwaves('C:\EEG_Data\Input', 'C:\EEG_Data\Output');
%
% % 自定义参数
% batch_process_slowwaves('C:\EEG_Data\Input', 'C:\EEG_Data\Output', ...
%     'freq_range', [0.5 4], ...
%     'amplitude_threshold', 80, ...
%     'duration_range', [0.5 2], ...
%     'channels', [1:32], ...
%     'detection_method', 'amplitude', ...
%     'time_window', [-0.5 1], ...
%     'separate_channels', false, ...
%     'file_pattern', '*.set', ...
%     'min_events', 5);
%
% % 分通道绘图
% batch_process_slowwaves('C:\EEG_Data\Input', 'C:\EEG_Data\Output', ...
%     'separate_channels', true, ...
%     'channels', [1:8]);