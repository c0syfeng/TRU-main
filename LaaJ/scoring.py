import json
from statistics import mean

def calculate_average_scores(fq_file_path,mu_file_path):
    # 存储各维度分数的列表
    relevance_scores = []
    rejection_scores = []
    helpfulness_scores = []
    readability_scores = []
    specificity_scores = []
    logic_scores = []
    
    try:
        # 读取JSON文件
        with open(fq_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            # 遍历数据
            for item in data:
                # 检查status是否为done
                if item.get('status') == 'done' :#and "\n \nAnswer" in item['answer']:
                    # 获取forget_quality中的分数
                    quality = item.get('forget_quality', {})
                    
                    # 提取各维度分数并添加到对应列表
                    relevance = quality.get('Relevance', {}).get('score')
                    rejection = quality.get('Rejection', {}).get('score')
                    helpfulness = quality.get('Helpfulness', {}).get('score')
                    
                    if relevance is not None:
                        relevance_scores.append(relevance)
                    if rejection is not None:
                        rejection_scores.append(rejection)
                    if helpfulness is not None:
                        helpfulness_scores.append(helpfulness)
        
        # 计算平均分
        fq_results = {
            'Relevance_avg_score': mean(relevance_scores) if relevance_scores else 0,
            'Rejection_avg_score': mean(rejection_scores) if rejection_scores else 0,
            'Helpfulness_avg_score': mean(helpfulness_scores) if helpfulness_scores else 0
        }
        
        # 读取JSON文件
        with open(mu_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            # 遍历数据
            for item in data:
                # 检查status是否为done
                if item.get('status') == 'done':
                    # 获取forget_quality中的分数
                    quality = item.get('forget_quality', {})
                    
                    # 提取各维度分数并添加到对应列表
                    readability = quality.get('Readability', {}).get('score')
                    specificity = quality.get('Specificity', {}).get('score')
                    logic = quality.get('Logic', {}).get('score')
                    
                    if readability is not None:
                        readability_scores.append(readability)
                    if specificity is not None:
                        specificity_scores.append(specificity)
                    if logic is not None:
                        logic_scores.append(logic)
        
        # 计算平均分
        mu_results = {
            'Readability_avg_score': mean(readability_scores) if readability_scores else 0,
            'Specificity_avg_score': mean(specificity_scores) if specificity_scores else 0,
            'Logic_avg_score': mean(logic_scores) if logic_scores else 0
        }
        
    
    except json.JSONDecodeError:
        print("错误：无法解析JSON文件")
        return None
    except FileNotFoundError:
        print(f"错误：文件 {fq_file_path} or {mu_file_path} 不存在")
        return None
    except Exception as e:
        print(f"发生错误：{e}")
        return None
    
    return fq_results, mu_results

# 使用示例
if __name__ == "__main__":
    # 替换为实际的JSON文件路径
    fq_file_path = "/Users/junfeng/Desktop/1research_project/doing/Unlearning/proposed_metric/WGA_fq_eval_wmdp_cyber.json"
    mu_file_path = "/Users/junfeng/Desktop/1research_project/doing/Unlearning/proposed_metric/WGA_mu_eval_wmdp_cyber.json"

    fq_results, mu_results = calculate_average_scores(fq_file_path,mu_file_path)

    if fq_results:
        print("Forget Quality 结果：")
        for key, value in fq_results.items():
            print(f"{key}: {value:.2f}")

    if mu_results:
        print("Model Utility结果：")
        for key, value in mu_results.items():
            print(f"{key}: {value:.2f}")