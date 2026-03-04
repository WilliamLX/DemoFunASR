import os
import requests
import json


def print_txt_filenames(directory):
    """Print the name of each .txt file in the given directory, one by one."""
    txt_files = [f for f in os.listdir(directory) if f.endswith(".txt")]
    print(f"Number of .txt files: {len(txt_files)}")
    return txt_files


def analyze_insurance_content(directory):
    """
    Analyze each .txt file in the directory using Ollama to check for insurance selling content.

    Args:
        directory (str): Path to the directory containing .txt files

    Returns:
        dict: Dictionary with filename as key and analysis result as value
    """
    txt_files = [f for f in os.listdir(directory) if f.endswith(".txt")]
    results = {}

    for filename in txt_files:
        file_path = os.path.join(directory, filename)

        try:
            # Read the content of the .txt file
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            # Prepare the prompt for Ollama
            prompt = f"""
            分析以下文本内容，判断是否包含与燃气保险销售相关的内容。
            查找与以下相关的关键词、短语或主题：
            - 保险政策
            - 保险销售
            - 保险代理人
            - 保险产品
            - 保险营销
            - 保险报价
            - 保险覆盖范围
            
            文本内容：
            {content}
            
            请用清晰的"是"或"否"回答，并简要说明你发现了什么或为什么确定它与保险销售无关。
            """

            print(prompt)
            # Make request to Ollama API
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "gemma3:4b",  # You can change this to your preferred model
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=30,
            )
            print(response)

            if response.status_code == 200:
                result = response.json()
                analysis = result.get("response", "No response from model")
                results[filename] = analysis
                print(f"Analyzed {filename}: {analysis}")
            else:
                results[filename] = f"Error: HTTP {response.status_code}"
                print(f"Error analyzing {filename}: HTTP {response.status_code}")

        except FileNotFoundError:
            results[filename] = "Error: File not found"
            print(f"Error: File {filename} not found")
        except Exception as e:
            results[filename] = f"Error: {str(e)}"
            print(f"Error analyzing {filename}: {str(e)}")

    return results


def print_insurance_analysis(directory):
    """
    Print a summary of insurance content analysis for all .txt files in the directory.

    Args:
        directory (str): Path to the directory containing .txt files
    """
    print(f"\n=== Insurance Content Analysis for {directory} ===")
    results = analyze_insurance_content(directory)

    print(f"\nSummary:")
    print(f"Total files analyzed: {len(results)}")

    insurance_files = []
    non_insurance_files = []

    for filename, analysis in results.items():
        if analysis.startswith("Error"):
            print(f"❌ {filename}: {analysis}")
        elif "YES" in analysis.upper():
            insurance_files.append(filename)
            print(f"✅ {filename}: Contains insurance-related content")
        else:
            non_insurance_files.append(filename)
            print(f"❌ {filename}: No insurance-related content found")

    print(f"\nFiles with insurance content: {len(insurance_files)}")
    print(f"Files without insurance content: {len(non_insurance_files)}")

    return results


# Example usage:
if __name__ == "__main__":
    txt_files = print_txt_filenames("data/text")
    print(txt_files)

    print_insurance_analysis("data/text")
