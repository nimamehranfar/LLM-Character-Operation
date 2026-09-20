from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from src.data.character_dataset import OPERATION_ARGUMENTS
from src.executor.operations import CharacterExecutor, CharacterOperation, ExecutionRequest

@dataclass(frozen=True)
class FinalPipelineExecution:
    operation: str
    arguments: dict[str,str]
    result: Any

def execute_parsed_call(parsed:dict[str,object],executor:CharacterExecutor)->FinalPipelineExecution:
    if parsed.get('decision')!='CALL' or not bool(parsed.get('valid')):raise ValueError('not an executable CALL')
    operation=str(parsed.get('operation','NONE'))
    if operation not in OPERATION_ARGUMENTS:raise ValueError(f'unsupported operation {operation!r}')
    raw={str(k):str(v) for k,v in dict(parsed.get('arguments',{})).items()};required=set(OPERATION_ARGUMENTS[operation])
    if set(raw)!=required:raise ValueError(f'{operation} arguments must be exactly {sorted(required)}')
    req=ExecutionRequest(operation=CharacterOperation(operation),text=raw['text'],character=raw.get('character'),index=int(raw['index']) if 'index' in raw else None,word_index=int(raw['word_index']) if 'word_index' in raw else None,insertion=raw.get('insertion'))
    return FinalPipelineExecution(operation=operation,arguments=raw,result=executor.execute(req))
