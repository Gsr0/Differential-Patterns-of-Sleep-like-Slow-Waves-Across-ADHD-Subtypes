import pandas as pd
import numpy as np
import os
import re
from pathlib import Path


class ANTDataExtractor:
    def __init__(self):
        """
        ANT数据提取器
        用于从E-Prime的.edat3文件或文本文件中提取ANT三个网络的指标
        """
        self.data = None
        self.results = None

    def parse_eprime_text(self, file_path):
        """
        解析E-Prime文本文件

        Parameters:
        -----------
        file_path : str
            文件路径

        Returns:
        --------
        pd.DataFrame : 解析后的数据框
        """
        data_list = []
        current_trial = {}

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()

        # 提取基本信息
        header_match = re.search(r'\*\*\* Header Start \*\*\*(.*?)\*\*\* Header End \*\*\*',
                                 content, re.DOTALL)

        subject_id = None
        session_date = None
        group = None

        if header_match:
            header_content = header_match.group(1)
            subject_match = re.search(r'Subject: (\w+)', header_content)
            date_match = re.search(r'SessionDate: ([\d-]+)', header_content)
            group_match = re.search(r'Group: (\d+)', header_content)

            if subject_match:
                subject_id = subject_match.group(1)
            if date_match:
                session_date = date_match.group(1)
            if group_match:
                group = int(group_match.group(1))

        # 提取所有LogFrame
        logframes = re.findall(r'\*\*\* LogFrame Start \*\*\*(.*?)\*\*\* LogFrame End \*\*\*',
                               content, re.DOTALL)

        for frame in logframes:
            trial_data = {
                'Subject': subject_id,
                'SessionDate': session_date,
                'Group': group
            }

            # 解析每行数据
            lines = frame.strip().split('\n')
            for line in lines:
                line = line.strip()
                if ':' in line and not line.startswith('Level'):
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()

                    # 转换数值
                    if value.isdigit():
                        value = int(value)
                    elif value.replace('.', '').isdigit():
                        value = float(value)

                    trial_data[key] = value

            # 只保留正式试验的数据（非练习试验）
            if trial_data.get('Procedure') != 'PRAC':
                data_list.append(trial_data)

        return pd.DataFrame(data_list)

    def load_edat3_file(self, file_path):
        """
        加载.edat3文件（需要先转换为文本格式）

        Parameters:
        -----------
        file_path : str
            .edat3文件路径
        """
        # 注意：.edat3文件通常需要E-DataAid软件转换为文本格式
        # 这里假设已经转换为文本格式
        print("注意：.edat3文件需要先用E-DataAid转换为文本格式")
        print("或者直接提供已转换的文本文件")

    def load_data(self, file_path):
        """
        加载数据文件

        Parameters:
        -----------
        file_path : str
            文件路径（支持.txt格式）
        """
        if file_path.endswith('.txt'):
            self.data = self.parse_eprime_text(file_path)
        elif file_path.endswith('.edat3'):
            print("请先将.edat3文件转换为文本格式，或使用E-DataAid导出为.txt文件")
            return None
        else:
            print("不支持的文件格式。请使用.txt或.edat3文件")
            return None

        print(f"成功加载数据，共 {len(self.data)} 个试验")
        return self.data

    def clean_data(self, min_rt=200, max_rt=2000):
        """
        清理数据

        Parameters:
        -----------
        min_rt : int
            最小反应时间（毫秒）
        max_rt : int
            最大反应时间（毫秒）
        """
        if self.data is None:
            print("请先加载数据")
            return

        original_count = len(self.data)

        # 移除缺失反应时间的试验
        self.data = self.data.dropna(subset=['Target.RT'])

        # 移除RT为0的试验（可能是未响应）
        self.data = self.data[self.data['Target.RT'] > 0]

        # 移除异常反应时间的试验
        self.data = self.data[
            (self.data['Target.RT'] >= min_rt) &
            (self.data['Target.RT'] <= max_rt)
            ]

        # 只保留正确试验用于RT分析（准确率分析需要所有试验）
        self.data_correct = self.data[self.data['Target.ACC'] == 1].copy()

        cleaned_count = len(self.data)
        print(f"数据清理完成：{original_count} -> {cleaned_count} 试验")
        print(f"正确试验数量：{len(self.data_correct)}")

    def calculate_ant_networks(self):
        """
        计算ANT三个网络的效应值

        Returns:
        --------
        dict : 包含三个网络效应值的字典
        """
        if self.data is None:
            print("请先加载数据")
            return None

        # 确保数据已清理
        if not hasattr(self, 'data_correct'):
            self.clean_data()

        # 按条件分组计算平均RT
        condition_rt = self.data_correct.groupby(['Cue', 'Flanker'])['Target.RT'].mean()

        results = {}

        try:
            # 1. 警觉网络效应 (Alerting Network)
            # 无提示条件 - 中央提示条件
            no_cue_rt = condition_rt.get(('No_Cue', 'Congruent'), 0) + \
                        condition_rt.get(('No_Cue', 'Incongruent'), 0) + \
                        condition_rt.get(('No_Cue', 'Neutral'), 0)
            no_cue_rt = no_cue_rt / 3  # 取平均

            center_cue_rt = condition_rt.get(('Center_Cue', 'Congruent'), 0) + \
                            condition_rt.get(('Center_Cue', 'Incongruent'), 0) + \
                            condition_rt.get(('Center_Cue', 'Neutral'), 0)
            center_cue_rt = center_cue_rt / 3  # 取平均

            alerting_effect = no_cue_rt - center_cue_rt
            results['Alerting_Effect'] = alerting_effect

            # 2. 定向网络效应 (Orienting Network)
            # 中央提示条件 - 空间提示条件
            spatial_cue_rt = condition_rt.get(('S_Cue', 'Congruent'), 0) + \
                             condition_rt.get(('S_Cue', 'Incongruent'), 0) + \
                             condition_rt.get(('S_Cue', 'Neutral'), 0)
            spatial_cue_rt = spatial_cue_rt / 3  # 取平均

            orienting_effect = center_cue_rt - spatial_cue_rt
            results['Orienting_Effect'] = orienting_effect

            # 3. 执行网络效应 (Executive Network)
            # 不一致条件 - 一致条件
            incongruent_rt = (condition_rt.get(('No_Cue', 'Incongruent'), 0) +
                              condition_rt.get(('Center_Cue', 'Incongruent'), 0) +
                              condition_rt.get(('S_Cue', 'Incongruent'), 0)) / 3

            congruent_rt = (condition_rt.get(('No_Cue', 'Congruent'), 0) +
                            condition_rt.get(('Center_Cue', 'Congruent'), 0) +
                            condition_rt.get(('S_Cue', 'Congruent'), 0)) / 3

            executive_effect = incongruent_rt - congruent_rt
            results['Executive_Effect'] = executive_effect

        except Exception as e:
            print(f"计算网络效应时出错: {e}")
            print("可用的条件组合:")
            print(condition_rt.index.tolist())

        return results

    def calculate_accuracy_rt(self):
        """
        计算总体准确率和反应时间

        Returns:
        --------
        dict : 包含准确率和反应时间指标的字典
        """
        if self.data is None:
            print("请先加载数据")
            return None

        results = {}

        # 总体准确率
        total_accuracy = self.data['Target.ACC'].mean()
        results['Overall_Accuracy'] = total_accuracy

        # 各条件准确率
        cue_accuracy = self.data.groupby('Cue')['Target.ACC'].mean()
        flanker_accuracy = self.data.groupby('Flanker')['Target.ACC'].mean()

        for cue_type, acc in cue_accuracy.items():
            results[f'{cue_type}_Accuracy'] = acc

        for flanker_type, acc in flanker_accuracy.items():
            results[f'{flanker_type}_Accuracy'] = acc

        # 反应时间（仅正确试验）
        if hasattr(self, 'data_correct'):
            overall_rt = self.data_correct['Target.RT'].mean()
            results['Overall_RT'] = overall_rt

            # 各条件反应时间
            cue_rt = self.data_correct.groupby('Cue')['Target.RT'].mean()
            flanker_rt = self.data_correct.groupby('Flanker')['Target.RT'].mean()

            for cue_type, rt in cue_rt.items():
                results[f'{cue_type}_RT'] = rt

            for flanker_type, rt in flanker_rt.items():
                results[f'{flanker_type}_RT'] = rt

        return results

    def extract_all_measures(self):
        """
        提取所有ANT指标

        Returns:
        --------
        dict : 包含所有指标的字典
        """
        if self.data is None:
            print("请先加载数据")
            return None

        # 基本信息
        subject_info = {
            'Subject': self.data['Subject'].iloc[0] if 'Subject' in self.data.columns else 'Unknown',
            'Group': self.data['Group'].iloc[0] if 'Group' in self.data.columns else 'Unknown',
            'SessionDate': self.data['SessionDate'].iloc[0] if 'SessionDate' in self.data.columns else 'Unknown'
        }

        # 网络效应
        network_effects = self.calculate_ant_networks()

        # 准确率和反应时间
        acc_rt_measures = self.calculate_accuracy_rt()

        # 合并所有结果
        all_results = {**subject_info, **network_effects, **acc_rt_measures}

        return all_results

    def process_multiple_files(self, folder_path, output_file='ant_results.csv'):
        """
        批量处理多个文件

        Parameters:
        -----------
        folder_path : str
            包含ANT数据文件的文件夹路径
        output_file : str
            输出CSV文件名
        """
        all_results = []

        # 查找所有文本文件
        folder = Path(folder_path)
        txt_files = list(folder.glob('*.txt'))

        print(f"找到 {len(txt_files)} 个文本文件")

        for file_path in txt_files:
            print(f"处理文件: {file_path.name}")

            try:
                # 加载数据
                self.load_data(str(file_path))

                # 提取指标
                results = self.extract_all_measures()

                if results:
                    all_results.append(results)
                    print(f"  成功提取数据")
                else:
                    print(f"  提取数据失败")

            except Exception as e:
                print(f"  处理文件时出错: {e}")

        # 保存结果
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(output_file, index=False)
            print(f"\n结果已保存到: {output_file}")
            print(f"共处理 {len(all_results)} 个被试的数据")

            # 显示结果摘要
            print("\n结果摘要:")
            print(results_df.describe())

            return results_df
        else:
            print("未能提取到任何数据")
            return None


# 使用示例
def main():
    """
    主函数 - 使用示例
    """
    # 创建提取器实例
    extractor = ANTDataExtractor()

    # 方法1: 处理单个文件
    print("=== 处理单个文件示例 ===")
    # file_path = "path/to/your/ant_data.txt"  # 替换为您的文件路径
    # extractor.load_data(file_path)
    # results = extractor.extract_all_measures()
    # print("提取的指标:")
    # for key, value in results.items():
    #     print(f"{key}: {value}")

    # 方法2: 批量处理多个文件
    print("=== 批量处理示例 ===")
    folder_path = "D:\桌面\正常儿童\\0525cyx\cyxbhv"  # 替换为您的数据文件夹路径
    results_df = extractor.process_multiple_files(folder_path, 'ant_network_results.csv')

    print("请将上述注释取消并修改文件路径后运行")


if __name__ == "__main__":
    main()