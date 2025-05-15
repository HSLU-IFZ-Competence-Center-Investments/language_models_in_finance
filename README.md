# Language Models in Finance

<img src="https://github.com/HSLU-IFZ-Competence-Center-Investments/Compliance_LM_2025/blob/main/language_models_in_finance/src/Images/Cover_01.jpg?raw=true" width="100%" />

## Initial Situation
The rise of artificial intelligence (AI) and natural language processing (NLP), particularly through language models (LMs), is enabling new ways for financial institutions to process information and make decisions more efficiently. In regulated sectors like finance, LMs offer the potential to automate tasks such as compliance monitoring while maintaining accuracy and ensuring data privacy. One key use case is the monitoring of Politically Exposed Persons (PEPs), where LMs can reduce manual effort and flag risks in real time by cross-referencing internal and external data. These capabilities help institutions meet regulatory standards, prevent fraud, and avoid reputational damage.

This repository represents an initial step toward integrating AI in banking by exploring how LMs can support key compliance processes, particularly in identifying individuals in positions of power within companies. It focuses on developing and testing a prototype that uses various LMs to automatically detect individuals in executive and board-level roles, which is a crucial step for effective oversight, influence mapping, and risk assessment.

## Prototype Description
The developed prototypes for automated leadership identification are built around two core modes: Mode A (LM-Only) and Mode B (Full Retrieval-Augmented Generation, RAG). The pipelines of these modes are illustrated in the figure below.

Mode A (LM-Only) does not incorporate any external information retrieval. Instead, the model relies entirely on its pre-trained internal knowledge. The corresponding pipeline follows these key steps:
1) Database: A name (e.g., "Jan Doe") is retrieved from a predefined dataset.
2) Prompt construction: A predefined instruction is combined with the name to form a minimal input prompt. The question posed: *"Is {name} a member of the management team (e.g. CEO, CFO, CTO, COO, CAO, CIO, EVP, head, director) or board of directors of a publicly listed company?"* The expected answer is either *Yes* or *No*.
3) LM: The model processes the prompt and responds with a binary decision *Yes* or *No* based only on its internal knowledge, without access to current web sources or external context.
4) Output logging: The model’s output and the corresponding input name are logged to ensure traceability, enabling later evaluation, analysis, and auditability of the inference results.

Mode B (Retrieval-Augmented Generation, RAG) employs a complete RAG pipeline and the corresponding pipeline follows these key steps:
1) Database: A name (e.g., "Jan Doe") is retrieved from a predefined dataset.
2) Google API: A query limited to information from the past year is executed using the input name as an exact match, and the ten most relevant web links are selected for further analysis.
3) Web scraping: The selected websites are parsed using *BeautifulSoup*, focusing strictly on HTML elements. Non-HTML content is excluded from parsing.
4) Context filtering: Each document represents content retrieved from a single URL. Paragraphs are selected as relevant if they contain part of the target name and at least one leadership-related keyword in German or English. The top paragraphs from each document are ranked and concatenated, with a character limit based on the model’s context window to ensure balanced and fair representation across documents. The final input is created by merging these truncated texts into a single context string.
5) Prompt construction: A predefined instruction is combined with the target name and the retrieved context to create the input prompt. The question posed: *"Is {name} a member of the management team (e.g. CEO, CFO, CTO, COO, CAO, CIO, EVP, head, director) or board of directors of a publicly listed company?"* The expected answer is either *Yes* or *No*.
6) LM: The model processes the prompt and responds with a binary decision *Yes* or *No* based on its internal knowledge and the context provided.
7) Output logging: The model’s output and the corresponding input name are logged to ensure traceability, enabling later evaluation, analysis, and auditability of the inference results.

<img src="https://github.com/HSLU-IFZ-Competence-Center-Investments/Compliance_LM_2025/blob/main/language_models_in_finance/src/Images/Prototypes_Modes.jpg?raw=true" width="100%" />

*Note: General architecture of the prototype.*

## Evaluation of the Prototype
To ensure a transparent and structured selection process, models were chosen based on four key criteria: ability to follow instructions, reasoning capabilities, context window size, and availability within the ecosystem. A range of models was selected to cover different capabilities, from efficient SLMs to powerful LLMs with advanced contextual reasoning. GPT-4.1 was included via the OpenAI’s API as a state-of-the-art benchmark, despite limitations in deployment control and openness. The selected models are shown in the table below:
| **Model**                         | **Parameters** | **Context Window** | **Provider**          |
|----------------------------------|----------------|---------------------|------------------------|
| TinyLlama-v1.1                   | 1.1B           | 2k tokens           | TinyLlama Team         |
| Phi-4-mini-instruct              | 3.84B          | 128k tokens         | Microsoft              |
| gemma-3-4b-it                    | 4.3B           | 128k tokens         | Google DeepMind        |
| zephyr-7b-beta                   | 7.24B          | 32k tokens          | Hugging Face H4 Team   |
| Mistral-7B-Instruct-v0.3         | 7.25B          | 32k tokens          | Mistral AI             |
| Llama-3.1-8B-Instruct            | 8.03B          | 128k tokens         | Meta AI                |
| DeepSeek-R1-Distill-Qwen-14B     | 14.8B          | 128k tokens         | DeepSeek AI            |
| Mixtral-8x22B Instruct v0.1      | 141B           | 64k tokens          | Mistral AI             |
| GPT-4.1                          | Undisclosed    | 1M tokens           | OpenAI                 |

*Note: Mixtral-8x22B is a mixture-of-experts model, activating only a subset of parameters per inference step.*

The evaluation methodology is illustrated in the figure below. To evaluate model performance, three real-world datasets of individuals in leadership roles at publicly listed companies were used: International, Swiss Large- and Mid-Cap, and Swiss Small-Cap. For each, a corresponding synthetic dataset with 100 randomly generated names was created by combining first and last names from the real dataset. Model performance was evaluated using standard classification outcomes: True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN). From these, key performance metrics (Precision, Recall, F1 Score, and Accuracy) were derived to assess classification performance.

<img src="https://github.com/HSLU-IFZ-Competence-Center-Investments/Compliance_LM_2025/blob/main/language_models_in_finance/src/Images/Analysis_Process.jpg?raw=true" width="100%" />

*Note: Evaluation process.*

## Testing the Prototype

To evaluate the prototype’s effectiveness in identifying individuals in leadership roles, we conducted an extensive benchmark across nine language models, comparing their performance in both Retrieval-Augmented Generation (RAG) and Non-RAG settings. The evaluation was based on three datasets:  
- **International large-cap companies**  
- **Swiss large- and mid-cap companies**  
- **Swiss small-cap companies**

All evaluations were carried out using a standardized pipeline without task-specific fine-tuning. The results provide insights into the relative strengths and limitations of various model architectures, as well as the impact of external context retrieval on classification accuracy.

### RAG vs. Non-RAG Mode

As illustrated in the figures below, enabling RAG consistently improved model performance across all datasets and models. This was especially evident in the international dataset, where rich and up-to-date online information allowed the models to make more accurate classifications. Smaller models like `TinyLlama` struggled to effectively utilize the retrieved context due to their limited capacity. In contrast, larger models such as `GPT-4.1` and `Mixtral-8x7B` achieved significant F1 score improvements with RAG enabled. Due to the clear performance gains, all further evaluations were conducted in RAG mode only.

<img src="https://github.com/HSLU-IFZ-Competence-Center-Investments/Compliance_LM_2025/blob/main/language_models_in_finance/src/plots/plots_f1_comparison_international_scope.png" width="100%" />

*Note: F1 Score comparison of various language models on the international dataset, with and without RAG.*

<img src="https://github.com/HSLU-IFZ-Competence-Center-Investments/Compliance_LM_2025/blob/main/language_models_in_finance/src/plots/plots_f1_comparison_switzerland_scope.png" width="100%" />

*Note: F1 Score comparison of various language models on the Switzerland large- and mid-cap dataset, with and without RAG.*

<img src="https://github.com/HSLU-IFZ-Competence-Center-Investments/Compliance_LM_2025/blob/main/language_models_in_finance/src/plots/plots_f1_comparison_switzerland_nebenwerte_scope.png" width="100%" />

*Note: F1 Score comparison of various language models on the Switzerland small-cap dataset, with and without RAG.*

### LM Comparison in RAG-Mode

Model performance varied considerably depending on the dataset. F1 scores were highest for the international dataset, reflecting the broader web visibility of leadership figures in global companies. Swiss large- and mid-cap firms followed closely, while Swiss small-cap companies posed the greatest challenge due to limited public data. Larger models (e.g., `DeepSeek`, `GPT-4.1`, `Mixtral`) consistently outperformed smaller ones, highlighting the role of model capacity in leveraging retrieved context effectively.

<img src="language_models_in_finance/src/Images/combined_radar_plots.jpg" width="100%" />

*Note: Comparison of different language models in terms of different data scopes.*

### False Negatives and Precision-Recall Trade-Offs

In compliance-related tasks, false negatives represent a significant risk. The following figures report the number of false negatives for the different models and datasets. Some models like `TinyLlama` reported low number of false negatives but at the expenses of precision, resulting in a higher number of false positives. Other models, such as `Zephyr` and `Llama-3.1`, were more conservative and thus missed more true positives. These results highlight the critical need to balance recall and precision when deploying such systems in operational environments.

<img src="https://github.com/HSLU-IFZ-Competence-Center-Investments/Compliance_LM_2025/blob/main/language_models_in_finance/src/plots/fn_comparison_international.png" width="100%" />

*Note: Comparison of how many false negatives were detected by each model on the international dataset, with RAG.*

<img src="https://github.com/HSLU-IFZ-Competence-Center-Investments/Compliance_LM_2025/blob/main/language_models_in_finance/src/plots/fn_comparison_switzerland.png" />

*Note: Comparison of how many false negatives were detected by each model on the Switzerland large- and mid-cap dataset , with RAG.*

<img src="https://github.com/HSLU-IFZ-Competence-Center-Investments/Compliance_LM_2025/blob/main/language_models_in_finance/src/plots/fn_comparison_switzerland_nebenwerte.png" />

*Note: Comparison of how many false negatives were detected by each model on the Switzerland small-cap dataset , with RAG.*

### Execution Time and Efficiency

Performance gains often came with higher computational costs. While larger models achieved higher F1 scores, they also required longer inference times. Interestingly, some models (e.g., `Phi-4-mini-instruct`) were able to achieve relatively strong performance with significantly lower latency, making them attractive options for real-time or resource-constrained use cases.

<img src="language_models_in_finance/src/Images/execution_time_vs_f1_rag_true_by_model_color.jpg" width="100%" />

*Note: An overview of the execution time of different models and data scopes, related to the model’s F1 score with
RAG enable.*

### Related Report
The report has been uploaded to the repository (add link) and is available publicly. 

