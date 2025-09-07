import pandas as pd


def analyze_ant_data(file_path='lht.txt'):
    """
    分析ANT任务的日志文件，提取行为学指标和注意网络效应值。

    参数:
    file_path (str): ANT任务日志文件（.txt格式）的路径。

    返回:
    dict: 包含所有计算指标的字典。
    """
    try:

        # 使用制表符作为分隔符加载数据
        df = pd.read_csv(file_path, sep='\t', on_bad_lines='warn')
        print(f"文件 '{file_path}' 加载成功，共包含 {len(df)} 行原始数据。")
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'。请确保文件路径正确。")
        return None

    # --- 1. 数据预处理 ---

    # 筛选出正式实验的试次 (排除练习 'PracProc')
    df_exp = df[df['Procedure'] == 'TrialProc'].copy()
    if df_exp.empty:
        print("错误：在文件中未找到正式实验试次（Procedure == 'TrialProc'）。请检查文件内容。")
        return None

    print(f"筛选出 {len(df_exp)} 行正式实验试次。")

    # 将反应时列转换为数值类型，错误时设置为NaN
    df_exp['Target.RT'] = pd.to_numeric(df_exp['Target.RT'], errors='coerce')

    # 清理数据：移除反应时过快 (<100ms) 或过慢 (>1500ms) 的试次
    # 注意：Target.RT == 0 表示漏判 (miss)，这里我们暂时保留
    valid_rt_mask = (df_exp['Target.RT'] >= 100) & (df_exp['Target.RT'] <= 1500)
    # 只有在有反应时的情况下才应用此过滤器
    responded_trials = df_exp[df_exp['Target.RT'] > 0].copy()
    responded_trials_cleaned = responded_trials[
        (responded_trials['Target.RT'] >= 100) & (responded_trials['Target.RT'] <= 1500)
        ]

    miss_trials = df_exp[df_exp['Target.RT'] == 0]

    print(f"在有反应的试次中，移除了 {len(responded_trials) - len(responded_trials_cleaned)} 个异常反应时试次。")

    # --- 2. 计算总体行为指标 ---

    total_trials = len(df_exp)
    num_responded = len(responded_trials_cleaned)
    num_miss = len(miss_trials) + (len(responded_trials) - len(responded_trials_cleaned))  # 漏判+移除的异常试次

    # 仅在有反应的、清理过的试次中计算正确率和错误率
    correct_trials_df = responded_trials_cleaned[responded_trials_cleaned['Target.ACC'] == 1]
    num_correct = len(correct_trials_df)
    num_error = num_responded - num_correct

    # 计算指标
    # 准确率 = 正确数 / (正确数 + 错误数)
    accuracy = num_correct / num_responded if num_responded > 0 else 0
    # 错误率 = 错误数 / (正确数 + 错误数)
    error_rate = num_error / num_responded if num_responded > 0 else 0
    # 漏判率 = 漏判数 / 总试次数
    miss_rate = num_miss / total_trials if total_trials > 0 else 0
    # 平均反应时（仅计算正确试次）
    mean_rt_correct = correct_trials_df['Target.RT'].mean()

    # --- 3. 计算注意网络效应值 ---

    # 定义计算各条件下平均反应时的函数（仅使用正确试次）
    def get_condition_rt(df_correct, cue_type=None, flank_type=None):
        df_slice = df_correct
        if cue_type:
            df_slice = df_slice[df_slice['CueType'] == cue_type]
        if flank_type:
            df_slice = df_slice[df_slice['FlankType'] == flank_type]
        return df_slice['Target.RT'].mean()

    # 计算各条件下的平均RT
    rt_no_cue = get_condition_rt(correct_trials_df, cue_type='nocue')
    rt_double_cue = get_condition_rt(correct_trials_df, cue_type='double')
    rt_center_cue = get_condition_rt(correct_trials_df, cue_type='center')
    rt_spatial_cue = get_condition_rt(correct_trials_df, cue_type='spatial')
    rt_congruent = get_condition_rt(correct_trials_df, flank_type='congruent')
    rt_incongruent = get_condition_rt(correct_trials_df, flank_type='incongruent')

    # 计算网络效应
    # 警觉效应 = 无线索RT - 双线索RT
    alerting_effect = rt_no_cue - rt_double_cue
    # 定向效应 = 中央线索RT - 空间线索RT
    orienting_effect = rt_center_cue - rt_spatial_cue
    # 执行控制(冲突)效应 = 不一致RT - 一致RT
    executive_control_effect = rt_incongruent - rt_congruent

    # --- 4. 结果汇总 ---

    results = {
        "总体准确率 (Accuracy)": f"{accuracy:.3f}",
        "总体错误率 (Error Rate)": f"{error_rate:.3f}",
        "总体漏判率 (Miss Rate)": f"{miss_rate:.3f}",
        "平均反应时 (正确试次, ms)": f"{mean_rt_correct:.2f}",
        "警觉网络效应 (Alerting, ms)": f"{alerting_effect:.2f}",
        "定向网络效应 (Orienting, ms)": f"{orienting_effect:.2f}",
        "执行控制效应 (Executive Control, ms)": f"{executive_control_effect:.2f}",
        "--- (详细计数) ---": "---",
        "总试次数": total_trials,
        "有效反应数": num_responded,
        "正确数": num_correct,
        "错误数": num_error,
        "漏判/剔除数": num_miss
    }

    return results


if __name__ == '__main__':
    # 调用函数并传入你的文件名
    file_path = 'lht.txt'
    """从单个ANT数据文件中提取行为指标"""
    with open(file_path, 'r', encoding='utf-16') as f:
        content = f.read()
    ant_results = analyze_ant_data(file_path)

    # 打印结果
    if ant_results:
        print("\n" + "=" * 40)
        print("      ANT 任务行为数据分析结果")
        print("=" * 40)
        for key, value in ant_results.items():
            print(f"{key:<40}: {value}")
        print("=" * 40)