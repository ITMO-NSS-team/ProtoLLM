import os
from pathlib import Path
from typing import List

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.metrics import ContextualRecallMetric
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.metrics import FaithfulnessMetric
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase
from deepeval.test_case import LLMTestCaseParams
import pandas as pd

from agents.agent import Agent
from agents.prompts import ac_cor_user_prompt
from agents.prompts import base_sys_prompt
from agents.prompts import binary_fc_user_prompt
from agents.prompts import fc_sys_prompt
from agents.prompts import fc_user_prompt
from agents.prompts import pip_cor_user_prompt
from agents.tools.accessibility_tools import accessibility_tools
from agents.tools.pipeline_tools import pipeline_tools
from modules.models.connector_creator import LanguageModelCreator
from modules.models.vsegpt_api import VseGPTConnector
from modules.variables import ROOT
from modules.variables.prompts import accessibility_sys_prompt
from modules.variables.prompts import strategy_sys_prompt
from pipelines.accessibility_pipeline import define_default_functions
from pipelines.accessibility_pipeline import set_default_value_if_empty
from pipelines.strategy_pipeline import retrieve_context_from_chroma
from pipelines.tests.utils_for_test import generate_path_to_results
from utils.measure_time import Timer


path_to_data = Path(ROOT, "pipelines", "tests")
# all_questions = pd.read_csv(Path(path_to_data, "test_data", "complete_dataset.csv"))
all_questions = pd.read_csv(Path(path_to_data, "test_data", "dataset_eng_full.csv"))
strategy_data = all_questions.loc[
    all_questions["correct_pipeline"] == "strategy_development_pipeline"
].reset_index()
access_data = all_questions.loc[
    all_questions["correct_pipeline"] == "service_accessibility_pipeline"
].reset_index()
# collection_name = "strategy-spb"
collection_name = "strategy-spb-eng"

model = VseGPTConnector(
    model="openai/gpt-4o-mini"
)  # possible to change model for metric evaluation

metrics_init_params = {
    "model": model,
    "verbose_mode": True,
    "async_mode": False,
}
answer_relevancy = AnswerRelevancyMetric(**metrics_init_params)
faithfulness = FaithfulnessMetric(**metrics_init_params)
context_precision = ContextualPrecisionMetric(**metrics_init_params)
context_recall = ContextualRecallMetric(**metrics_init_params)
context_relevancy = ContextualRelevancyMetric(**metrics_init_params)
correctness_metric = GEval(
    name="Correctness",
    criteria=(
        "1. Correctness and Relevance:"
        "- Compare the actual response against the expected response. Determine the"
        " extent to which the actual response captures the key elements and concepts of"
        " the expected response."
        "- Assign higher scores to actual responses that accurately reflect the core"
        " information of the expected response, even if only partial."
        "2. Numerical Accuracy and Interpretation:"
        "- Pay particular attention to any numerical values present in the expected"
        " response. Verify that these values are correctly included in the actual"
        " response and accurately interpreted within the context."
        "- Ensure that units of measurement, scales, and numerical relationships are"
        " preserved and correctly conveyed."
        "3. Allowance for Partial Information:"
        "- Do not heavily penalize the actual response for incompleteness if it covers"
        " significant aspects of the expected response. Prioritize the correctness of"
        " provided information over total completeness."
        "4. Handling of Extraneous Information:"
        "- While additional information not present in the expected response should not"
        " necessarily reduce score, ensure that such additions do not introduce"
        " inaccuracies or deviate from the context of the expected response."
    ),
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    **metrics_init_params,
)


def choose_pipeline_test(answer_check: bool = False) -> pd.DataFrame:
    """Tests pipeline choosing functions.

    Counts number of correctly chosen pipelines and measures elapsed time for
    choosing and checking results.
    Writes results to .txt file at the specified path.

    Args:
        answer_check: flag whether to do a response check (True) or not (False)

    Returns: None
    """
    print("Pipeline choosing test is running...")
    test_name = "pipeline_selection"
    model_name = os.getenv("MODEL_NAME")
    path_to_results, path_to_df = generate_path_to_results(
        test_name, model_name, answer_check
    )
    os.makedirs(os.path.dirname(path_to_results), exist_ok=True)

    results = {
        "question": [],
        "territory_name": [],
        "correct_pipeline": [],
        "pipeline_from_model": [],
        "pipeline_selection_time": [],
    }

    for i, row in all_questions.iterrows():
        print(f"Processing question {i}")
        question = row["question"].replace('"', "'")
        correct_pipeline = row["correct_pipeline"]
        t_n = row["territory_name"]
        if pd.isnull(t_n):
            t_name = None
        else:
            try:
                t_name = int(t_n)
            except ValueError:
                t_name = t_n
        results["question"].append(question)
        results["correct_pipeline"].append(correct_pipeline)
        results["territory_name"].append(t_name)
        agent = Agent("LLAMA_FC_URL", pipeline_tools)
        with Timer() as t:
            try:
                pipeline = agent.choose_functions(
                    question, fc_sys_prompt, binary_fc_user_prompt
                )
                results["pipeline_from_model"].append(pipeline[0])
            except:
                results["pipeline_from_model"].append(None)
            results["pipeline_selection_time"].append(t.seconds_from_start)
        if answer_check:
            with Timer() as t:
                if results.get("pipeline_checking_time") is None:
                    results["pipeline_checking_time"] = []
                    results["checked_pipeline_from_model"] = []
                try:
                    checked_pipeline = agent.check_functions(
                        question, pipeline, base_sys_prompt, pip_cor_user_prompt
                    )
                    results["checked_pipeline_from_model"].append(checked_pipeline[0])
                except:
                    results["checked_pipeline_from_model"].append(None)
                results["pipeline_checking_time"].append(t.seconds_from_start)

    result_df = pd.DataFrame.from_dict(results)
    result_df.to_csv(path_to_df, index=False)
    # Add a calculation of the percentage of correct pipelines with and without validation
    result_df.loc[:, "correct_or_not"] = result_df.apply(
        lambda r: 1 if r["pipeline_from_model"] == r["correct_pipeline"] else 0, axis=1
    )
    corr_pipe_percent = (
        result_df["correct_or_not"].value_counts(normalize=True).loc[1].round(4) * 100
    )
    avg_pipe_choose_time = result_df["pipeline_selection_time"].mean().round(2)
    avg_pipe_check_time = "RUN WITHOUT CHECKING"
    corr_pipe_percent_check = "RUN WITHOUT CHECKING"
    if answer_check:
        result_df.loc[:, "correct_or_not_check"] = result_df.apply(
            lambda r: 1
            if r["checked_pipeline_from_model"] == r["correct_pipeline"]
            else 0,
            axis=1,
        )
        corr_pipe_percent_check = (
            result_df["correct_or_not_check"].value_counts(normalize=True).loc[1].round(4)
            * 100
        )
        avg_pipe_check_time = result_df["pipeline_checking_time"].mean().round(2)

    to_print = f"""Percentage of correctly chosen pipeline: {corr_pipe_percent}
Percentage of correctly chosen pipeline with checking: {corr_pipe_percent_check}
Average pipeline choosing time: {avg_pipe_choose_time}
Average pipeline checking time: {avg_pipe_check_time}"""

    with open(path_to_results, "w") as f:
        print(to_print, file=f)

    return result_df


# The same metrics as this function are counted in the next test, so there is little point
# in running it separately
def choose_functions_test(answer_check: bool = False) -> None:
    """Tests API functions choosing function.

    Counts number of correctly chosen function and measures elapsed time for
    choosing and checking results.
    Writes results to .txt file at the specified path.

    Args:
        answer_check: flag whether to do a response check (True) or not (False)

    Returns: None
    """
    print("API functions choosing test is running...")
    test_name = "function_selection"
    model_name = os.getenv("MODEL_NAME")
    path_to_results, path_to_df = generate_path_to_results(
        test_name, model_name, answer_check
    )
    os.makedirs(os.path.dirname(path_to_results), exist_ok=True)

    results = {
        "question": [],
        "correct_function": [],
        "function_from_model": [],
        "function_selection_time": [],
    }

    for i, row in access_data.iterrows():
        print(f"Processing question {i}")
        question = row["question"].replace('"', "'")
        correct_function = row["correct_functions"]
        results["question"].append(question)
        results["correct_function"].append(correct_function)
        t_t = row["territory_type"]
        t_n = row["territory_name"]
        cs = row["geometry"]
        t_type = None if pd.isnull(t_t) else t_t
        if pd.isnull(t_n):
            t_name = None
        else:
            try:
                t_name = int(t_n)
            except ValueError:
                t_name = t_n
        if pd.isnull(cs):
            coordinates = None
        else:
            coordinates = eval(cs)["coordinates"]
        agent = Agent("LLAMA_FC_URL", accessibility_tools)
        with Timer() as t:
            functions = agent.choose_functions(question, fc_sys_prompt, fc_user_prompt)
            results["function_selection_time"].append(t.seconds_from_start)
        if answer_check:
            with Timer() as t:
                functions = agent.check_functions(
                    question, functions, base_sys_prompt, ac_cor_user_prompt
                )
                if results.get("function_checking_time") is None:
                    results["function_checking_time"] = []
                    results["function_checking_time"].append(t.seconds_from_start)
                else:
                    results["function_checking_time"].append(t.seconds_from_start)
        functions = functions + define_default_functions(t_type, t_name, coordinates)
        functions = set_default_value_if_empty(functions)
        results["function_from_model"].append(functions)

    result_df = pd.DataFrame.from_dict(results)
    result_df.to_csv(path_to_df, index=False)
    result_df.loc[:, "correct_or_not"] = result_df.apply(
        lambda r: 1 if r["correct_function"] in r["function_from_model"] else 0, axis=1
    )
    corr_func = (
        result_df["correct_or_not"].value_counts(normalize=True).loc[1].round(4) * 100
    )
    avg_func_choose_time = result_df["function_selection_time"].mean().round(2)
    avg_func_check_time = "RUN WITHOUT CHECKING"
    if answer_check:
        avg_func_check_time = result_df["function_checking_time"].mean().round(2)

    to_print = f"""Percentage of correctly chosen functions: {corr_func}
Average function choosing time: {avg_func_choose_time}
Average function checking time: {avg_func_check_time}"""

    with open(path_to_results, "w") as f:
        print(to_print, file=f)


def accessibility_pipeline_test(
    metrics_to_calculate: List, answer_check: bool = False
) -> pd.DataFrame:
    """Tests whole accessibility pipeline.

    Counts the number of correctly chosen functions and correct answers,
    measures elapsed time for choosing and checking and final answer
    generation. Writes results to .txt file at the specified path.

    Args:
        metrics_to_calculate: list of metrics to be calculated
        answer_check: flag whether to do a response check (True) or not (False)

    Returns: None
    """
    print("Accessibility pipeline test is running...")
    test_name = "accessibility_pipeline"
    model_name = os.getenv("MODEL_NAME")
    path_to_results, path_to_df = generate_path_to_results(
        test_name, model_name, answer_check
    )
    os.makedirs(os.path.dirname(path_to_results), exist_ok=True)

    results = {
        "question": [],
        "territory_name": [],
        "correct_pipeline": [],
        "correct_answer": [],
        "correct_function": [],
        "function_from_model": [],
        "functions_for_context": [],
        "context": [],
        "answer_from_model": [],
        "function_selection_time": [],
        "answer_generation_time": [],
    }
    for metric in metrics_to_calculate:
        results[f"{metric.__name__}_score"] = []
        results[f"{metric.__name__}_reason"] = []

    for i, row in all_questions.iterrows():
        print(f"Processing question {i}")
        question = row["question"].replace('"', "'")
        correct_pipeline = row["correct_pipeline"]
        correct_function = row["correct_functions"]
        correct_answer = row["correct_answer"]
        t_t = row["territory_type"]
        t_n = row["territory_name"]
        cs = row["geometry"]
        t_type = None if pd.isnull(t_t) else t_t
        if pd.isnull(t_n):
            t_name = None
        else:
            try:
                t_name = int(t_n)
            except ValueError:
                t_name = t_n
        if pd.isnull(cs):
            coordinates = None
        else:
            coordinates = eval(cs)["coordinates"]
        results["question"].append(question)
        results["correct_function"].append(correct_function)
        results["correct_answer"].append(correct_answer)
        results["correct_pipeline"].append(correct_pipeline)
        results["territory_name"].append(t_name)
        agent = Agent("LLAMA_FC_URL", accessibility_tools)
        with Timer() as t:
            try:
                functions = agent.choose_functions(
                    question, fc_sys_prompt, fc_user_prompt
                )
            except:
                functions = []
            results["function_selection_time"].append(t.seconds_from_start)
            results["function_from_model"].append(functions)
            functions = define_default_functions(t_type, t_name, coordinates) + functions
            functions = set_default_value_if_empty(functions)
            results["functions_for_context"].append(functions)
        if answer_check:
            if results.get("function_checking_time") is None:
                results["function_checking_time"] = []
                results["checked_function_from_model"] = []
                results["functions_for_context_check"] = []
            with Timer() as t:
                try:
                    functions = agent.check_functions(
                        question, functions, base_sys_prompt, ac_cor_user_prompt
                    )
                except:
                    functions = []
                results["function_checking_time"].append(t.seconds_from_start)
                results["checked_function_from_model"].append(functions)
                functions = (
                    define_default_functions(t_type, t_name, coordinates) + functions
                )
                functions = set_default_value_if_empty(functions)
                results["functions_for_context_check"].append(functions)
        try:
            context = agent.retrieve_context_from_api(
                t_name, t_type, coordinates, functions
            )
        except:
            context = ""
        results["context"].append(context)
        with Timer() as t:
            model_url = os.environ.get("LLAMA_URL")
            model_connector = LanguageModelCreator.create_llm_connector(
                model_url, accessibility_sys_prompt
            )
            try:
                llm_res = model_connector.generate(question, context)
                results["answer_from_model"].append(llm_res)
            except:
                llm_res = "No response due to an error"
                results["answer_from_model"].append(None)
            results["answer_generation_time"].append(t.seconds_from_start)
        test_case = LLMTestCase(
            input=question,
            actual_output=llm_res,
            expected_output=correct_answer,
            retrieval_context=[context],
        )
        for metric in metrics_to_calculate:
            try:
                metric.measure(test_case)
                results[f"{metric.__name__}_score"].append(metric.score)
                results[f"{metric.__name__}_reason"].append(metric.reason)
            except Exception as e:
                results[f"{metric.__name__}_score"].append("-1")
                results[f"{metric.__name__}_reason"].append(
                    type(e).__name__ + " - " + str(e)
                )
                continue

    result_df = pd.DataFrame.from_dict(results)
    result_df["total_time"] = (
        result_df["function_selection_time"] + result_df["answer_generation_time"]
    )
    if answer_check:
        result_df["total_time"] = (
            result_df["total_time"] + result_df["function_checking_time"]
        )
    result_df.to_csv(path_to_df, index=False)
    mask = result_df["correct_pipeline"] == "service_accessibility_pipeline"
    correct_or_not = result_df[mask].apply(
        lambda r: 1 if r["correct_function"] in r["functions_for_context"] else 0, axis=1
    )
    # Calculation of basic statistics for exec time and function selection
    corr_func = correct_or_not.value_counts(normalize=True).loc[1].round(4) * 100
    avg_func_choose_time = result_df["function_selection_time"].mean().round(2)
    corr_func_check = "RUN WITHOUT CHECKING"
    avg_func_check_time = "RUN WITHOUT CHECKING"
    if answer_check:
        correct_or_not_check = result_df[mask].apply(
            lambda r: 1
            if r["correct_function"] in r["functions_for_context_check"]
            else 0,
            axis=1,
        )
        corr_func_check = (
            correct_or_not_check.value_counts(normalize=True).loc[1].round(4) * 100
        )
        avg_func_check_time = result_df["function_checking_time"].mean().round(2)
    avg_ans_generation_time = result_df["answer_generation_time"].mean().round(2)
    avg_total_time = result_df["total_time"].mean().round(2)
    # Calculation of statistics for metrics
    metrics_score_columns = list(
        filter(lambda x: "score" in x, result_df.columns.tolist())
    )
    metrics_to_print = []
    for column in metrics_score_columns:
        result_df[column] = pd.to_numeric(result_df[column])
        avg_score = result_df[result_df[column] != -1][column].mean()
        failed_evaluations = result_df[result_df[column] == -1].shape[0]
        metrics_to_print.append(
            f"Average {column} is {avg_score}. Number of unsuccessfully"
            f" processed questions {failed_evaluations}"
        )
    short_metrics_result = "\n".join(metrics_to_print)

    to_print = f"""Percentage of correctly chosen functions: {corr_func}
Percentage of correctly chosen functions with checking: {corr_func_check}
Average function choosing time: {avg_func_choose_time}
Average functions checking time: {avg_func_check_time}
Average answer generation time: {avg_ans_generation_time}
Average total time: {avg_total_time}
Short metrics results:
{short_metrics_result}"""

    with open(path_to_results, "w") as f:
        print(to_print, file=f)

    return result_df


def strategy_pipeline_test(
    metrics_to_calculate: List, chunk_num: int = 4
) -> pd.DataFrame:
    """Tests strategy pipeline.

    Evaluate metrics for model answers using 'deepeval' and measures elapsed
    time for context retrieving and final answer generation.
    Writes results to .txt file at the specified path.

    Args:
        metrics_to_calculate: list of metrics to be calculated
        chunk_num: number of chunks to be extracted from vector storage

    Returns: None
    """
    print("Strategy pipeline test is running...")
    test_name = "strategy_pipeline"
    model_name = os.getenv("MODEL_NAME")
    path_to_results, path_to_df = generate_path_to_results(test_name, model_name)
    os.makedirs(os.path.dirname(path_to_results), exist_ok=True)

    results = {
        "question": [],
        "territory_name": [],
        "correct_answer": [],
        "context": [],
        "answer_from_model": [],
        "answer_generation_time": [],
    }
    for metric in metrics_to_calculate:
        results[f"{metric.__name__}_score"] = []
        results[f"{metric.__name__}_reason"] = []

    for i, row in all_questions.iterrows():
        print(f"Processing question {i}")
        question = row["question"].replace('"', "'")
        correct_answer = row["correct_answer"]
        t_n = row["territory_name"]
        if pd.isnull(t_n):
            t_name = None
        else:
            try:
                t_name = int(t_n)
            except ValueError:
                t_name = t_n
        results["question"].append(question)
        results["correct_answer"].append(correct_answer)
        results["territory_name"].append(t_name)
        context = retrieve_context_from_chroma(question, collection_name, chunk_num)
        results["context"].append(context)
        with Timer() as t:
            model_url = os.environ.get("LLAMA_URL")
            context = context.replace('"', "'")
            model_connector = LanguageModelCreator.create_llm_connector(
                model_url, strategy_sys_prompt
            )
            try:
                llm_ans = model_connector.generate(question, context)
                results["answer_from_model"].append(llm_ans)
            except:
                llm_ans = "No response due to an error"
                results["answer_from_model"].append(None)
            results["answer_generation_time"].append(t.seconds_from_start)
        test_case = LLMTestCase(
            input=question,
            actual_output=llm_ans,
            expected_output=correct_answer,
            retrieval_context=[context],
        )
        for metric in metrics_to_calculate:
            try:
                metric.measure(test_case)
                results[f"{metric.__name__}_score"].append(metric.score)
                results[f"{metric.__name__}_reason"].append(metric.reason)
            except Exception as e:
                results[f"{metric.__name__}_score"].append("-1")
                results[f"{metric.__name__}_reason"].append(
                    type(e).__name__ + " - " + str(e)
                )
                continue

    result_df = pd.DataFrame.from_dict(results)
    result_df["total_time"] = result_df["answer_generation_time"]
    result_df.to_csv(path_to_df, index=False)
    # Calculation of basic statistics for exec time
    avg_total_time = result_df["total_time"].mean().round(2)
    metrics_score_columns = list(
        filter(lambda x: "score" in x, result_df.columns.tolist())
    )
    metrics_to_print = []
    for column in metrics_score_columns:
        result_df[column] = pd.to_numeric(result_df[column])
        avg_score = result_df[result_df[column] != -1][column].mean()
        failed_evaluations = result_df[result_df[column] == -1].shape[0]
        metrics_to_print.append(
            f"Average {column} is {avg_score}. Number of unsuccessfully"
            f" processed questions {failed_evaluations}"
        )
    short_metrics_result = "\n".join(metrics_to_print)

    to_print = f"""Average total time (answer generation time): \
{avg_total_time}
Short metrics results:
{short_metrics_result}"""

    with open(path_to_results, "w") as f:
        print(to_print, file=f)

    return result_df


def complete_pipeline_metrics(
    pipeline_selection: str = None,
    accessibility_pipeline: str = None,
    strategy_pipeline: str = None,
    check_flag: bool = False,
) -> None:
    """Calculate complete statistics for app

    Args:
        pipeline_selection: pipeline selection test results file
        accessibility_pipeline: accessibility pipeline test results file
        strategy_pipeline: strategy pipeline test results file
        check_flag: run tests with answer check or not
    """
    files = [pipeline_selection, accessibility_pipeline, strategy_pipeline]
    test_names = ["pipeline_selection", "accessibility_pipeline", "strategy_pipeline"]
    paths = []
    if not all(files):
        pipeline_selection_df = choose_pipeline_test(answer_check=check_flag)
        access_pipeline_df = accessibility_pipeline_test(metrics, answer_check=check_flag)
        strategy_pipeline_df = strategy_pipeline_test(metrics, chunks)
    else:
        for test_name, file_name in zip(test_names, files):
            paths.append(Path(path_to_data, "test_results", test_name, file_name))
        pipeline_selection_df = pd.read_csv(paths[0])
        access_pipeline_df = pd.read_csv(paths[1])
        strategy_pipeline_df = pd.read_csv(paths[2])

    metrics_results = dict()

    for pipeline in ["service_accessibility_pipeline", "strategy_development_pipeline"]:
        if check_flag:
            selected_pipeline = pipeline_selection_df[
                pipeline_selection_df["checked_pipeline_from_model"] == pipeline
            ]
        else:
            selected_pipeline = pipeline_selection_df[
                pipeline_selection_df["pipeline_from_model"] == pipeline
            ]

        if pipeline == "service_accessibility_pipeline":
            df_to_calculate = selected_pipeline.merge(
                access_pipeline_df,
                on=["question", "territory_name"],
                how="inner",
                suffixes=["_1", "_2"],
            )
        else:
            df_to_calculate = selected_pipeline.merge(
                strategy_pipeline_df,
                on=["question", "territory_name"],
                how="inner",
                suffixes=["_1", "_2"],
            )
        mask = (df_to_calculate["answer_from_model"].isnull()) | (
            df_to_calculate["answer_from_model"] == ""
        )
        failed_questions = df_to_calculate[mask].shape[0]
        avg_ans_generation_time = (
            df_to_calculate["answer_generation_time"].mean().round(2)
        )
        avg_total_time = df_to_calculate["total_time"].mean().round(2)
        metrics_score_columns = list(
            filter(lambda x: "score" in x, df_to_calculate.columns.tolist())
        )
        metrics_to_print = []
        for column in metrics_score_columns:
            df_to_calculate[column] = pd.to_numeric(df_to_calculate[column])
            failed_evaluations = df_to_calculate[df_to_calculate[column] == -1].shape[0]
            avg_score = df_to_calculate[df_to_calculate[column] != -1][column].mean()
            if metrics_results.get(column) is None:
                metrics_results[column] = []
            metrics_results[column].append(avg_score)
            metrics_to_print.append(
                f"Average {column} is {avg_score}. Number of unsuccessfully"
                f" processed questions {failed_evaluations}"
            )
        short_metrics_result = "\n".join(metrics_to_print)
        print(f"""Results for questions that have been assigned to the {pipeline}
Number of failed questions: {failed_questions}
Average answer generation time ({pipeline}): {avg_ans_generation_time}
Average total time: ({pipeline}): {avg_total_time}
Metrics results:
{short_metrics_result}
""")

    model_name = os.getenv("MODEL_NAME")
    print(f"Results for {model_name}")
    for k in metrics_results.keys():
        avg = sum(metrics_results[k]) / len(metrics_results[k])
        print(f"Average result for {k} is {avg}")


if __name__ == "__main__":
    chunks = 4
    # metrics = [answer_relevancy, faithfulness, correctness_metric]
    metrics = [correctness_metric, answer_relevancy]
    # metrics = []
    # choose_pipeline_test(answer_check=True)
    # accessibility_pipeline_test(metrics, answer_check=True)
    # strategy_pipeline_test(metrics, chunks)

    # path_1 = "pipeline_selection_with_check_mixtral-8x22b-instruct_2024-10-09 19:00:00.csv"
    # path_2 = "accessibility_pipeline_with_check_mixtral-8x22b-instruct_2024-10-09 19:11:00.csv"
    # path_3 = "strategy_pipeline_mixtral-8x22b-instruct_2024-10-09-13:59:00.csv"
    # complete_pipeline_metrics(path_1, path_2, path_3, check_flag=True)

    complete_pipeline_metrics(check_flag=False)