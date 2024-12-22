from typing import List
import json
import copy
import logging
from samplefactory.synthetic_pipelines.prompts import (generate_summary_system_prompt, generate_summary_evaluation_system_prompt,
                                                        generate_rag_system_prompt, check_summary_quality_human_prompt,
                                                        generate_rag_human_prompt, generate_aspect_summarisation_prompt,
                                                        generate_summary_human_prompt, generate_aspect_summarisation_evaluation_system_prompt,
                                                        generate_quiz_system_prompt, generate_quiz_human_prompt)
from samplefactory.utils import Dataset
import numpy as np
import asyncio
from typing import List, Optional, Dict, Any, TypeVar, Union, cast

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import RetryOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (Runnable, RunnablePassthrough, 
                                      RunnableParallel, RunnableLambda)
from langchain.chains.combine_documents import create_stuff_documents_chain
from openai import APIConnectionError
from samplefactory.synthetic_pipelines.genetic_evolver import GeneticEvolver

import random
from samplefactory.synthetic_pipelines.schemes import (SummaryQualitySchema, 
                                                       RAGScheme, AspectSummarisationQualitySchema,
                                                       QuizScheme, FreeQueryScheme)

import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
TT = TypeVar("TT")

class BaseChain:
    def __init__(self, llm: BaseChatModel, max_retry_attempts: int = 3):
        self._max_retry_attempts = max_retry_attempts
        self.llm = llm
        self.data = None

    def _enhance_prompt_with_cot(self, prompt: str) -> str:
        return (
            prompt +
            "\nLet's approach the task step by step:"
            "\n1. First, analyze the important elements for the task."
            "\n2. Then, analyze their interconnections."
            "\n3. Finally, form the conclusion in the required format."
            "\n\nThe response should be in JSON format according to the schema.\n\n"
        )
    
    def _enhance_prompt_with_tot(self, prompt: str) -> str:
        return (
            prompt +
            "\nConsider several solutions before forming the final conclusion. "
            "Ensure the final answer matches the specified JSON format."
        )

    def _prepare_prompt(self, system_template, human_template, x: Dict[str, Any]) -> PromptValue:
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template([{"text": system_template}]),
                HumanMessagePromptTemplate.from_template([{"text": human_template}])
            ]
        )

        args = x.copy()
        prompt_value = prompt.invoke(args)
        return prompt_value

    async def _run_chain_with_retries(self,
                                      chain: Runnable,
                                      chain_kwargs: Dict[str, Any],
                                      max_retry_attempts: Optional[int] = None) -> Optional[Any]:
        retry_attempts = max_retry_attempts or self._max_retry_attempts
        retry_delay = 2
        attempt = 0
        result = None

        while attempt < retry_attempts:
            try:
                result = await chain.ainvoke(chain_kwargs)
            except OutputParserException:
                logger.warning("Parsing error occurred. Interrupting execution.")
                raise
            except APIConnectionError:
                logger.warning(
                    f"Encountered APIConnectionError. Attempt: {attempt}. Will retry after delay {retry_delay} seconds",
                    exc_info=True
                )
                await asyncio.sleep(retry_delay)
            except Exception as e:
                logger.warning(f"Unexpected error during chain execution: {str(e)}", exc_info=False)
                
            attempt += 1

        return result

    def create_chain(self,
                          system_template: str,
                          human_template: str,
                          parser: Optional[PydanticOutputParser] = None,
                          max_parsing_retries: int = 3,
                          use_chain_of_thought: bool = False,
                          use_tree_of_thought: bool = False) -> Runnable:
        """Helper method to create chains that process text documents"""
        if use_chain_of_thought:
            human_template = self._enhance_prompt_with_cot(human_template)
        
        if use_tree_of_thought:
            system_template = self._enhance_prompt_with_tot(system_template)

        prompt = RunnableLambda(lambda x: self._prepare_prompt(system_template, human_template, x))

        chain = (
            prompt
            | RunnableParallel(completion=self.llm, prompt_value=RunnablePassthrough())
        )

        if parser:
            retry_planner_parser = RetryOutputParser.from_llm(
                parser=parser,
                llm=self.llm,
                prompt=PromptTemplate.from_template("{prompt}"),
                max_retries=max_parsing_retries
            )

            def _do_parsing_retrying(x: dict):
                result = None
                completion = x['completion'].content
                prompt_value = x['prompt_value']

                try:
                    result = retry_planner_parser.parse_with_prompt(completion=completion, prompt_value=prompt_value)
                except OutputParserException:
                    prompt_value = cast(PromptValue, x['prompt_value'])
                    logger.warning("Proceeding without result due to parser errors (even after retrying). "
                                   "Prompt - %s" % prompt_value)

                return result

            chain = (
                RunnableLambda(lambda x: {**x, "response_format_description": parser.get_format_instructions()})
                | chain
                | RunnableLambda(_do_parsing_retrying, name="retry_planner_lambda")
            )

        return chain

    def save_chain_output(self, path: str):
        if 'generated' in self.data.columns:  
            with open(path, 'w') as f:
                json.dump(self.data, f)
        else:
            logger.warning("No output to save. Chain has not been run.")

    def _sample_data(self, n_samples: int):
        """Sample a specified number of examples from the DataFrame."""
        if self.data is not None:
            n_samples = min(n_samples, len(self.data))
            self.data = self.data.sample(n=n_samples, random_state=random.randint(0, 10000)).reset_index(drop=True)  # Ensure we do not sample more than available
        else:
            logger.warning("No data to sample. Chain has not been run.")


class SummarisationChain(BaseChain):
    def __init__(self, llm: Optional[BaseChatModel] = None):
        super().__init__(llm=llm)

    async def _generate_summaries(self, data: List[str]) -> List[str]:
        chain = self.create_chain(system_template=generate_summary_system_prompt(), 
                                  human_template=generate_summary_human_prompt(),)
        summaries = await chain.abatch(inputs=[{'text': i} for i in data])
        return summaries

    async def _evaluate_summary_quality(self, summary: List[str], text: List[str]) -> List[SummaryQualitySchema]:
        parser = PydanticOutputParser(pydantic_object=SummaryQualitySchema)
        chain = self.create_chain(
            system_template=generate_summary_evaluation_system_prompt(), 
            human_template=check_summary_quality_human_prompt(), 
            parser=parser
        )
        
        # TODO: Use _run_chain_with_retries to handle retries
        response = await chain.abatch(inputs=[{'text': i, 'summary': j} for i, j in zip(text, summary)])
        return response
    
    async def postprocess(self, summaries: List[str], data: List[str]) -> Dataset:
        quality = await self._evaluate_summary_quality(summaries, data)
        logger.info(f"Summary Quality: {np.mean([int(sample.score) for sample in quality])}")
        # select only data indices with quality score above 0.8
        good_indices = [i for i in range(len(quality)) if quality[i].score > 0.8]
        return good_indices
    
    async def run(self, 
            dataset: Dataset, 
            n_examples: int = 3,
            do_postprocess: bool = True
            ) -> str:
        data = dataset.data[dataset.data_col].tolist()
        # sampling without changing the order of the data
        idx = random.sample(range(len(data)), n_examples)
        data = [data[i] for i in sorted(idx)]
        logger.info(f"Running summarisation chain on {n_examples} examples")
        summaries = await self._generate_summaries(data)
        summaries = [i['completion'].content for i in summaries]
        dataset.data['generated'] = summaries
        self.generated = dataset.data
        logger.info(f"Summary examples: {random.sample(summaries, 1)}")
        if do_postprocess:
            good_indices = await self.postprocess(summaries, data)
            dataset.data = dataset.data.iloc[good_indices]
        return summaries
    
# TODO: Implement aspect summarisation chain
class AspectSummarisationChain(BaseChain):
    def __init__(self, llm: BaseChatModel):
        if llm is None:
            raise ValueError("LLM must be provided for AspectSummarisationChain")
        super().__init__(llm=llm)

    async def _generate_aspect_summaries(self, 
                                         data: List[str], 
                                         aspect: str) -> List[str]:
        chain = self.create_chain(system_template=generate_aspect_summarisation_prompt(aspect=aspect), 
                                  human_template=generate_summary_human_prompt(),)
        summaries = await chain.abatch(inputs=[{'text': i} for i in data])
        return summaries

    async def _evaluate_summary_quality(self, summary: List[str], text: List[str], aspect: str) -> List[SummaryQualitySchema]:
        parser = PydanticOutputParser(pydantic_object=AspectSummarisationQualitySchema)
        chain = self.create_chain(
            system_template=generate_aspect_summarisation_evaluation_system_prompt(aspect=aspect), 
            human_template=check_summary_quality_human_prompt(), 
            parser=parser
        )
        
        # TODO: Use _run_chain_with_retries to handle retries
        response = await chain.abatch(inputs=[{'text': i, 'summary': j} for i, j in zip(text, summary)])

        return response

    async def run(self, 
                  dataset: Dataset, 
                  aspect: str,
                  n_examples: int = 3) -> str:
        data = self._sample_data(dataset.data, n_examples)  # Use the new sampling method
        logger.info(f"Running aspect summarisation chain on {n_examples} examples")
        summaries = await self._generate_aspect_summaries(data[dataset.data_col].tolist(), aspect=aspect)
        summaries = [i['completion'].content for i in summaries]
        dataset.data['generated'] = summaries
        self.generated = dataset.data
        logger.info(f"Summary examples: {random.sample(summaries, 1)}")
        quality = await self._evaluate_summary_quality(summaries, data[dataset.data_col].tolist(), aspect=aspect)
        logger.info(f"Summary Quality: {np.mean([int(sample.score) for sample in quality])}")
        return summaries
    
    # TODO: Implement postprocessing of the dataset
    def postprocess(self, dataset: Dataset) -> Dataset:
        dataset.data['generated'] = [i['completion'].content for i in self.generated]
        return dataset

class RAGChain(BaseChain):
    def __init__(self, llm: BaseChatModel):
        if llm is None:
            raise ValueError("LLM must be provided for RAGChain")
        super().__init__(llm=llm)

    async def _generate_rag_examples(self, data: List[str]) -> str:
        parser = PydanticOutputParser(pydantic_object=RAGScheme)
        chain = self.create_chain(
            system_template=generate_rag_system_prompt(), 
            human_template=generate_rag_human_prompt(),
            parser=parser
        )
        rag_examples = await chain.abatch(inputs=[{'text': i} for i in data])
        return rag_examples

    async def run(self, 
                  dataset: Dataset, 
                  n_examples: int = 3) -> str:
        self.data = copy.deepcopy(dataset.data)
        self._sample_data(n_samples=n_examples)  # Use the new sampling method
        logger.info(f"Running RAG chain on {n_examples} examples")
        rag_examples = await self._generate_rag_examples(self.data[dataset.data_col].tolist())
        logger.info(f"RAG examples: {random.sample(rag_examples, 1)}")
        print(len(rag_examples), len(self.data))
        self.data['generated'] = [[{'question': rag_examples[i].context[j].question, 
                                      'answer': rag_examples[i].context[j].answer, 
                                      'difficulty': rag_examples[i].context[j].difficulty} for j in range(len(rag_examples[i].context))
                                      ] for i in range(len(rag_examples))]
        return rag_examples


class QuizChain(BaseChain):
    def __init__(self, llm: Optional[BaseChatModel] = None):
        super().__init__(llm=llm)


    async def _generate_quiz(self, dataset: Dataset) -> str:
        parser = PydanticOutputParser(pydantic_object=QuizScheme)
        chain = self.create_chain(system_template=generate_quiz_system_prompt(), 
                                  human_template=generate_quiz_human_prompt(),
                                  parser=parser)
        quiz = await chain.abatch(inputs=[{'text': i} for i in dataset.data[dataset.data_col].tolist()])
        return quiz
    
    def postprocess(self, dataset: pd.DataFrame) -> Dataset:
        raise NotImplementedError("Not implemented")
    
    async def run(self, 
                  dataset: Dataset, 
                  n_examples: int = 3,
                  do_postprocess: bool = False
                  ) -> str:
        self.data = copy.deepcopy(dataset.data)
        self._sample_data(n_samples=n_examples)
        logger.info(f"Running quiz chain on {n_examples} examples")
        quiz = await self._generate_quiz(self.data[dataset.data_col].tolist())
        self.data['generated'] = quiz
        if do_postprocess:
            dataset = self.postprocess(self.data)
        return quiz

class FreeQueryChain(BaseChain):
    def __init__(self, llm: Optional[BaseChatModel] = None):
        super().__init__(llm=llm)

    def _generate_free_query_instruction(self, dataset: Dataset) -> List[FreeQueryScheme]:
        dataset.labeled_data
        raise NotImplementedError("Not implemented")
    
    def _combine_free_query_instruction(self, dataset: Dataset) -> str:
        raise NotImplementedError("Not implemented")

    def run(self, dataset: Dataset, n_examples: int = 3, 
            initial_instruction: str = None) -> str:
        self.data = copy.deepcopy(dataset.data)
        self._sample_data(n_samples=n_examples)
        logger.info(f"Running free query chain on {n_examples} examples: Instruction Extraction")
        free_query_instruction = self._generate_free_query_instruction(self.data, initial_instruction)

        return free_query_instruction


class GeneticEvolverChain(BaseChain):
    def __init__(self, llm: Optional[BaseChatModel] = None):
        super().__init__(llm=llm)

    def run(self, text: str) -> str:

        raise NotImplementedError("Not implemented")