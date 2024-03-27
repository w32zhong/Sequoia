export CUDA_VISIBLE_DEVICES=0,1

#python testbed_greedy.py --model  JackFram/llama-68m   --target NousResearch/Llama-2-7b-chat-hf  --T 0.6 --P 1.0  --start 0 --end 2 --M 384 --growmap ../L40_growmaps/68m_7b/growmaps/L40-CNN-68m-7b-greedy.pt  --Mode greedy --dataset cnn
python testbed_greedy.py --model  JackFram/llama-68m   --target NousResearch/Llama-2-7b-chat-hf  --T 0.6 --P 1.0  --start 0 --end 2 --M 384 --growmap ../L40_growmaps/68m_7b/growmaps/L40-OpenWebText-68m-7b-greedy.pt  --Mode greedy --dataset cnn

python testbed_greedy.py --model  JackFram/llama-68m   --target NousResearch/Llama-2-7b-chat-hf  --T 0.6 --P 1.0  --start 0 --end 2 --M 384 --growmap ../L40_growmaps/68m_7b/growmaps/L40-OpenWebText-68m-7b-greedy.pt  --Mode baseline --dataset cnn
