#!/bin/sh

#sbatch --mem=128GB --cpus-per-task=32 ./scripts/train.sh --batch_size=4096 --vocab_size=1048576 --reg_l1=1e-3 --output_dir=output/big_l1=1e-3
#sbatch --mem=128GB --cpus-per-task=32 ./scripts/train.sh --batch_size=4096 --vocab_size=1048576 --reg_l1=1 --output_dir=output/big_l1=1
#sbatch --mem=16GB --cpus-per-task=32 ./scripts/train.sh --batch_size=4096 --vocab_size=131072 --reg_l1=1 --output_dir=output/small_l1=1

#sbatch --mem=16GB --cpus-per-task=32 ./scripts/train.sh --batch_size=4096 --vocab_size=131072 --num_embeddings=1 --output_dir=output/small_ne=1

#sbatch --mem=20GB --cpus-per-task=4 -p gpu --gres=gpu:1 ./scripts/train.sh --batch_size=128 --vocab_size=32768 --num_embeddings=1 --nce_samples=256 --data_source=crawls --output_dir=output/crawls_e1
#sbatch --mem=20GB --cpus-per-task=4 -p gpu --gres=gpu:1 ./scripts/train.sh --batch_size=128 --vocab_size=32768 --num_embeddings=3 --nce_samples=256 --data_source=crawls --output_dir=output/crawls_e3
#sbatch --mem=20GB --cpus-per-task=4 -p gpu --gres=gpu:1 ./scripts/train.sh --batch_size=128 --vocab_size=32768 --num_embeddings=3 --nce_samples=256 --data_source=crawls --reg_l1=1 --output_dir=output/crawls_e3_l1

#sbatch --mem=20GB --cpus-per-task=4 -p gpu --gres=gpu:1 ./scripts/train.sh --batch_size=128 --vocab_size=131072 --num_embeddings=5 --nce_samples=256 --data_source=crawls --output_dir=output/crawls_e5_131072
#sbatch --mem=20GB --cpus-per-task=4 -p gpu --gres=gpu:1 ./scripts/train.sh --batch_size=128 --vocab_size=131072 --num_embeddings=5 --nce_samples=256 --data_source=crawls --reg_l1=0.001 --output_dir=output/crawls_e5_131072_l1=1e-3
#sbatch --mem=20GB --cpus-per-task=4 -p gpu --gres=gpu:1 ./scripts/train.sh --batch_size=128 --vocab_size=131072 --num_embeddings=5 --nce_samples=256 --data_source=crawls --reg_l1=1 --output_dir=output/crawls_e5_131072_l1=1

sbatch --mem=10GB --cpus-per-task=4 -p gpu --gres=gpu:1 ./scripts/train.sh --batch_size=128 --vocab_size=32768 --num_embeddings=5 --nce_samples=256 --data_source=crawls --output_dir=output/crawls_e5_32768
sbatch --mem=10GB --cpus-per-task=4 -p gpu --gres=gpu:1 ./scripts/train.sh --batch_size=128 --vocab_size=32768 --num_embeddings=5 --nce_samples=256 --data_source=crawls --reg_l1_diff=1e-3 --output_dir=output/crawls_e5_32768_l1diff=1e-3
sbatch --mem=10GB --cpus-per-task=4 -p gpu --gres=gpu:1 ./scripts/train.sh --batch_size=128 --vocab_size=32768 --num_embeddings=5 --nce_samples=256 --data_source=crawls --reg_l1_diff=1 --output_dir=output/crawls_e5_32768_l1diff=1
sbatch --mem=10GB --cpus-per-task=4 -p gpu --gres=gpu:1 ./scripts/train.sh --batch_size=128 --vocab_size=32768 --num_embeddings=5 --nce_samples=256 --data_source=crawls --reg_l2_diff=1e-3 --output_dir=output/crawls_e5_32768_l2diff=1e-3
