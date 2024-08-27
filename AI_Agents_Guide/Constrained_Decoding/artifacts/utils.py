import json
import typing
from collections import defaultdict
from typing import DefaultDict, Dict

import torch
from lmformatenforcer import JsonSchemaParser, TokenEnforcer
from lmformatenforcer.integrations.trtllm import build_trtlmm_tokenizer_data
from outlines.fsm.guide import RegexGuide
from outlines.fsm.json_schema import build_regex_from_schema
from outlines.integrations.utils import adapt_tokenizer
from pydantic import BaseModel
from transformers import AutoTokenizer


class WandFormat(BaseModel):
    wood: str
    core: str
    length: float


class AnswerFormat(BaseModel):
    name: str
    house: str
    blood_status: str
    occupation: str
    alive: str
    wand: WandFormat


class LMFELogitsProcessor:
    PROCESSOR_NAME = "lmfe"

    def __init__(self, tokenizer_dir, schema):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, legacy=False, padding_side="left", trust_remote_code=True
        )
        self.eos_token = tokenizer.eos_token_id
        tokenizer_data = build_trtlmm_tokenizer_data(tokenizer)
        self.token_enforcer = TokenEnforcer(tokenizer_data, JsonSchemaParser(schema))

    def get_allowed_tokens(self, ids):
        def _trim(ids):
            return [x for x in ids if x != self.eos_token]

        allowed = self.token_enforcer.get_allowed_tokens(_trim(ids[0]))
        return allowed

    def __call__(
        self,
        req_id: int,
        logits: torch.Tensor,
        ids: typing.List[typing.List[int]],
        stream_ptr: int,
    ):
        mask = torch.full_like(logits, fill_value=float("-inf"), device=logits.device)
        allowed = self.get_allowed_tokens(ids)
        mask[:, :, allowed] = 0
        with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
            logits += mask


class OutlinesLogitsProcessor:
    PROCESSOR_NAME = "outlines"

    def __init__(self, tokenizer_dir, schema):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, legacy=False, padding_side="left", trust_remote_code=True
        )
        tokenizer = adapt_tokenizer(tokenizer)
        regex_string = build_regex_from_schema(json.dumps(schema))
        self.fsm = RegexGuide(regex_string, tokenizer)
        self._fsm_state: DefaultDict[int, int] = defaultdict(int)
        self.mask_cache: Dict[int, torch.Tensor] = {}
        self._prefix = [-1]

    def __call__(
        self,
        req_id: int,
        logits: torch.Tensor,
        ids: typing.List[typing.List[int]],
        stream_ptr: int,
    ):
        # Initialize the FSM state dictionary if the input_ids are empty, as this means
        # that the input_ids are the first tokens of the sequence.
        seq_id = hash(tuple(ids[0]))
        # If the prefix token IDs have changed we assume that we are dealing with a new
        # sample and reset the FSM state
        if ids[0][: len(self._prefix)] != self._prefix:
            self._fsm_state = defaultdict(int)
            self._prefix = ids[0]
            seq_id = hash(tuple([]))

        else:
            # Remove the prefix token IDs from the input token IDs, as the FSM should
            # only be applied to the generated tokens
            ids = ids[0][len(self._prefix) :]
            last_token = ids[-1]
            last_seq_id = hash(tuple(ids[:-1]))
            seq_id = hash(tuple(ids))
            self._fsm_state[seq_id] = self.fsm.get_next_state(
                state=self._fsm_state[last_seq_id], token_id=last_token
            )

        state_id = self._fsm_state[seq_id]
        if state_id not in self.mask_cache:
            allowed_tokens = self.fsm.get_next_instruction(
                state=self._fsm_state[seq_id]
            ).tokens
            mask = torch.full_like(
                logits, fill_value=float("-inf"), device=logits.device
            )
            mask[:, :, allowed_tokens] = 0
            self.mask_cache[state_id] = mask
        else:
            mask = self.mask_cache[state_id]

        with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
            logits += mask
