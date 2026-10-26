import copy
import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[2]/"scripts/temporal_mixing.py"
spec=importlib.util.spec_from_file_location("temporal_mixing",path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixture():
    roots=[dict(root_id=f"r{i}",binding="task",workload=w,arrival=i*.2,target=i==0)
           for i,w in enumerate(["loop","loop","spawn","loop","join","loop"])]
    bank={"task/"+w:{"operations":[{"llm":{"input_token_count":n,"recorded_output_length":3},"dependencies":[]}]}
          for w,n in [("loop",2),("spawn",5),("join",8)]}
    return roots,bank


def test_deterministic_mapping_preserves_work_and_targets():
    roots,bank=fixture();before=copy.deepcopy(roots)
    block=module.make_block(roots,bank)
    assert block==module.make_block(roots,bank)
    assert roots==before and block["audit"]["passed"]
    assert block["interleaved"]!=block["clustered"]


@pytest.mark.parametrize("field,value",[("arrival",10),("binding","other"),("workload","join"),("root_id","r99")])
def test_controls_detect_changed_target_or_population(field,value):
    roots,bank=fixture();other=copy.deepcopy(roots);other[0][field]=value
    with pytest.raises((AssertionError,KeyError)):
        module.audit(roots,other,bank)


def test_controls_detect_changed_background_slots():
    roots,bank=fixture();other=copy.deepcopy(roots);other[1]["arrival"]+=1
    with pytest.raises(AssertionError,match="slots"):
        module.audit(roots,other,bank)
