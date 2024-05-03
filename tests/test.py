import sys
sys.path.append("..")
import torch
from transformers import AutoTokenizer

from Tree.SpecTree import SpecTree
from Engine.Engine import GraphInferenceEngine, GraphInferenceEngineTG
from utils import cuda_graph_for_residual, cuda_graph_for_sampling_without_replacement

model = 'JackFram/llama-68m'
target = 'NousResearch/Llama-2-7b-chat-hf'
max_length = 900

def prepare():
    T = 0.6
    top_p = 1.
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    draft_model = GraphInferenceEngine(max_length=max_length, model_name_or_path=model, dtype=torch.float16, device="cuda:0")
    target_model =  GraphInferenceEngineTG(max_length=max_length, model_name_or_path=target, dtype=torch.float16, device="cuda:0")

    residual_graph = cuda_graph_for_residual()
    path = '../L40_growmaps/68m_7b/growmaps/L40-CNN-68m-7b-stochastic.pt'
    grow_map = torch.load(path)

    tree_size = grow_map["size"]
    idx_lists = grow_map["roots"]
    branch_lists = grow_map['branches']
    draft_step = len(grow_map["roots"])
    graph_capture_list = [sum(x) for x in branch_lists]
    graph_capture_list.append(1)
    draft_model.initialize_cuda_graph(graph_capture_list)
    sampling_callables = {}
    sample_gather_indices = {}
    for i in range(draft_step - 1):
        idx_len = len(idx_lists[i])
        num_samples = max(branch_lists[i])
        sampling_callables[i] = cuda_graph_for_sampling_without_replacement(
            max_length=max_length, idx_len=idx_len, num_samples=num_samples,
            temperature=T, tree_size=tree_size) 
    for i in range(draft_step - 1):
        ith_gather_list = []
        max_num_samples = max(branch_lists[i])
        for j, branch in enumerate(branch_lists[i]):
            branch_index = torch.arange(branch, device="cuda:0", dtype=torch.long)
            branch_index = branch_index + j * max_num_samples
            ith_gather_list.append(branch_index)
        ith_gather_list = torch.cat(ith_gather_list)
        sample_gather_indices[i] = ith_gather_list
    return tokenizer, (target_model, draft_model), dict(T=T, top_p=top_p,
        max_length=max_length, residual_graph=residual_graph, grow_map=grow_map,
        sampling_callables=sampling_callables, sample_gather_indices=sample_gather_indices
    )

def simulation_fast(input_ids, target_model, draft_model, tokenizer, **kwargs):
    T = kwargs.pop('T')
    max_length = kwargs['max_length']
    dtype = torch.float16
    attn_mask = torch.full((max_length, max_length), torch.finfo(dtype).min, dtype=dtype, device='cuda:0')
    sequence = torch.tensor(list(range(max_length)), device='cuda:0').long().unsqueeze(-1)
    new_tokens_buffer = torch.zeros(max_length).long().to('cuda:0')
    parents_buffer = torch.zeros(max_length).long().to('cuda:0')
    position_ids = torch.zeros(max_length).long().to('cuda:0')
    draft_kv_len = 0
    target_kv_len = 0
    attn_mask.fill_(torch.finfo(dtype).min)
    spectree = SpecTree(prefix=input_ids.squeeze(0), device='cuda:0', temperature=T,
        draft_kv_len=draft_kv_len, target_kv_len=target_kv_len,
        draft_model_engine=draft_model, target_model_engine=target_model,
        position_ids=position_ids, attn_mask=attn_mask, sequence=sequence,
        parents_buffer=parents_buffer, new_tokens_buffer=new_tokens_buffer, 
        **kwargs)

    torch.cuda.synchronize()
    terminate = False
    prev_len = input_ids.shape[-1]
    prompt_len = input_ids.shape[-1]
    while input_ids.shape[1] < max_length and terminate == False:
        spectree.construct_grow_map()
        valid_tokens, draft_kv_len, target_kv_len, terminate = spectree.verify()
        input_ids = valid_tokens.unsqueeze(0)
        print(tokenizer.decode(input_ids[0][prev_len:]), flush=True, end=' ')
        prev_len = input_ids.shape[-1]
        if (input_ids[0][-1] == tokenizer.eos_token_id) or (input_ids[0][-1] == tokenizer.unk_token_id):
            terminate = True
    print()
    torch.cuda.synchronize()
    draft_model.clear_kv()
    target_model.clear_kv()

tokenizer, (target_model, draft_model), kwargs = prepare()

prompt = '[INST] what is 1 + 1? [/INST]'
input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to('cuda:0')
with torch.no_grad():
    simulation_fast(input_ids, target_model, draft_model, tokenizer, **kwargs)

prompt = '[INST] what is 2 + 2? [/INST]'
input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to('cuda:0')
with torch.no_grad():
    simulation_fast(input_ids, target_model, draft_model, tokenizer, **kwargs)
