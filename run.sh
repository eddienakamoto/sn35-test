# Run base model vllm
. vllm/bin/activate
pm2 start "vllm serve Qwen/Qwen2-7B-Instruct --port 30479 --host 0.0.0.0" --name "sn35-vllm" # change port and host to your preference


pm2 start "vllm serve edwardnakamoto/sn35-qwen-v1 --port 40569 --host 0.0.0.0" --name "sn35-vllm" # change port and host to your preference
pm2 start "vllm serve edwardnakamoto/autotrain-5yyjx-x8l7u --port 40569 --host 0.0.0.0" --name "sn35-vllm" # change port and host to your preference

writer: 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