"""
A job's adapter (Job(lora = ...)) and the prefix cache: pages are hashed with a salt per adapter, so the same
tokens cached with one adapter are never found by a job running with another, while the base model and jobs that
name no adapter hash as they always did. On CPU without a model.
"""

from unittest.mock import patch

import pytest
import torch

from exllamav3.constants import PAGE_SIZE
from exllamav3.generator.job import Job
from exllamav3.generator.pagetable import Sequence, lora_cache_salt, tensor_hash_checksum
from exllamav3.modules.linear import ALL_LORAS

pytestmark = pytest.mark.nogpu


class Adapter:
    pass


def page_hashes(ids, lora = ALL_LORAS):
    job = Job(input_ids = ids, max_new_tokens = 8, lora = lora)
    seq = job.sequences[0]
    seq.prepare(False, 8)
    return seq.page_hashes


def test_salt_per_adapter():
    a, b = Adapter(), Adapter()
    assert lora_cache_salt(None) is None and lora_cache_salt(ALL_LORAS) is None
    assert len(lora_cache_salt(a)) == 16
    assert lora_cache_salt(a) == lora_cache_salt(a) != lora_cache_salt(b)


def test_cache_key_stands_for_the_weights():
    a, b, c = Adapter(), Adapter(), Adapter()
    a.cache_key, b.cache_key, c.cache_key = "voice-1", b"voice-1", "voice-2"
    assert lora_cache_salt(a) == lora_cache_salt(b) != lora_cache_salt(c)
    assert lora_cache_salt(a) != lora_cache_salt(Adapter())


def test_an_adapter_named_by_a_plain_value_is_its_own_key():
    """Tensors may be registered under a name: no weak reference to a str, and the same name is the same adapter"""
    assert lora_cache_salt("voice-1") == lora_cache_salt("".join(["voice", "-1"])) != lora_cache_salt("voice-2")
    assert lora_cache_salt(b"voice-1") != lora_cache_salt("voice-1") != lora_cache_salt(1)
    assert Job(input_ids = torch.tensor([[1, 2, 3]]), lora = "voice-1").sequences[0].salt == lora_cache_salt("voice-1")


def test_an_object_that_takes_no_weak_reference_keeps_its_key_on_itself():
    """Nothing read off the object would do: its address can be another adapter's later"""
    class Slotted:
        __slots__ = ("cache_key",)

    class Bare:
        __slots__ = ()

    first, second = Slotted(), Slotted()
    assert lora_cache_salt(first) == lora_cache_salt(first) != lora_cache_salt(second)
    with pytest.raises(TypeError, match = "cache_key"):
        lora_cache_salt(Bare())


def test_all_loras_is_no_value_an_adapter_could_have():
    import copy
    import pickle
    assert ALL_LORAS != "all" and repr(ALL_LORAS) == "ALL_LORAS"
    assert pickle.loads(pickle.dumps(ALL_LORAS)) is ALL_LORAS and copy.deepcopy(ALL_LORAS) is ALL_LORAS
    job = Job(input_ids = torch.tensor([[1, 2, 3]]), lora = "".join(["a", "ll"]))    # an adapter named "all"
    assert job.lora is not ALL_LORAS and job.sequences[0].salt == lora_cache_salt("all") is not None


def test_pages_hash_apart_per_adapter_and_as_before_without_one():
    ids = torch.arange(2 * PAGE_SIZE + 5).view(1, -1)
    a, b = Adapter(), Adapter()
    plain = page_hashes(ids)
    assert len(plain) == 2
    assert plain[0] == tensor_hash_checksum(ids[:, :PAGE_SIZE], None)             # unsalted: the hash it always had
    assert page_hashes(ids, None) == plain
    with_a, with_b = page_hashes(ids, a), page_hashes(ids, b)
    assert with_a == page_hashes(ids, a)
    assert not set(with_a) & set(with_b) and not set(with_a) & set(plain)


def test_requeue_keeps_the_adapter():
    a = Adapter()
    job = Job(input_ids = torch.tensor([[1, 2, 3]]), max_new_tokens = 1000, max_rq_tokens = 256, lora = a)
    job.cached_pages, job.cached_tokens = 0, 0
    job.new_tokens = 250
    job.sequences[0].sequence_ids.append(torch.zeros((1, 250), dtype = torch.long))
    with patch.object(Job, "prepare_for_queue"):
        job = job.prepare_for_requeue()
    assert job.lora is a and job.sequences[0].salt == lora_cache_salt(a)


def test_default_job_names_no_adapter():
    job = Job(input_ids = torch.tensor([[1, 2, 3]]))
    assert job.lora is ALL_LORAS and job.sequences[0].salt is None
    assert Job(input_ids = torch.tensor([[1, 2, 3]]), lora = None).lora is None
