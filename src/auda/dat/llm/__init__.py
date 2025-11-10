from typing import List

from auda.utils.llm import ChatgptLocal, ChatgptRemote, LargeLanguageModel

from .__common import LLM_KIND, LlmISName, LlmOSName

llm_list: List[LargeLanguageModel] = [ChatgptRemote(), ChatgptLocal()]

__all__ = ['llm_list', 'LlmISName', 'LlmOSName', 'LLM_KIND']
