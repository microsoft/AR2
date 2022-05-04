

# Training instruction for Wikipedia NQ & TriviaQA

## Requirements
- torch==1.7.1+cu110
- transformers==4.7.0
- apex==0.1
- faiss==1.7.1
- tensorboardX==2.4.1

## Resources
### Data

|  Filename  |Note |
|  ----      |----  |
|[psgs_w100.tsv](https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz)| Wikipedia passages pool|
| [biencoder-trivia-train.json.gz](https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-train.json.gz)  | TriviaQA train subset  |
| [biencoder-trivia-dev.json.gz](https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-dev.json.gz)  | TriviaQA dev subset  |
| [trivia-train.qa.csv.gz](https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-train.qa.csv.gz)  | TriviaQA train subset for validation. (Query, Answer)  |
| [trivia-dev.qa.csv.gz](https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-dev.qa.csv.gz)  | TriviaQA dev subset for validation. (Query, Answer)  |
| [trivia-test.qa.csv.gz](https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-test.qa.csv.gz)  | TriviaQA test subset for validation. (Query, Answer)  |
| [biencoder-nq-train.json.gz](https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz)  | NQ train subset  |
| [biencoder-nq-dev.json.gz](https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz)  | NQ dev subset  |
| [nq-train.qa.csv](https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-train.qa.csv)  | TriviaQA train subset for validation. (Query, Answer)  |
| [nq-dev.qa.csv](https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-dev.qa.csv)  | TriviaQA dev subset for validation. (Query, Answer)  |
| [nq-test.qa.csv](https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-test.qa.csv)  | NQ test subset for validation. (Query, Answer)  |

### CKPT

|  Filename  |Note |
|  ----      |----  |
|[wiki_ict.pt](https://msraprophetnet.blob.core.windows.net/ar2/realease_ckpt/wiki_ict.pt)| The ckpt after ICT training  |
|[nq_fintinue.pkl](https://msraprophetnet.blob.core.windows.net/ar2/realease_ckpt/nq_fintinue.pkl)| The retriever ckpt after AR2 training on NQ |
|[triviaqa_fintinue.pkl](https://msraprophetnet.blob.core.windows.net/ar2/realease_ckpt/triviaqa_fintinue.pkl)| The retriever ckpt after AR2 training on TriviaQA |

## TriviaQA

We use TriviaQA as an example to show the process.

### 1. Warm retriever 

```bash
mkdir ../output
mkdir ../tensorboard_log

EXP_NAME=run_de_ict_triviaqa   # de means dual encoder.
DATA_DIR=/mnt/data/denseIR/data/trivia_data/
Model_After_ICT=../wiki_ict.pt  # CKPT initialization, where we use ict-trained.  
OUT_DIR=../output/$EXP_NAME
TB_DIR=../tensorboard_log/$EXP_NAME    # tensorboard log path

python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=9539 \
./wiki/run_de_model_ernie.py \
--model_type="nghuyong/ernie-2.0-en" \
--origin_data_dir=$DATA_DIR/biencoder-trivia-train.json \
--origin_data_dir_dev=$DATA_DIR/biencoder-trivia-dev.json \
--model_name_or_path_ict=$Model_After_ICT \
--max_seq_length=256 --per_gpu_train_batch_size=16 --gradient_accumulation_steps=1 \
--learning_rate=2e-5 --output_dir $OUT_DIR \
--warmup_steps 4000 --logging_steps 100 --save_steps 1000 --max_steps 40000 \
--log_dir $TB_DIR --fp16 \
--number_neg 1
```

### 2. Evaluate retriever and generate hard topk

```bash
EXP_NAME=run_de_ict_triviaqa              # de means dual encoder.
DATA_DIR=/mnt/data/denseIR/data/trivia_data/
CKPT_NUM=40000
OUT_DIR=../output/$EXP_NAME

python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=9539 \
./wiki/inference_de_wiki_gpu.py \
--model_type="nghuyong/ernie-2.0-en" \
--eval_model_dir=$OUT_DIR/checkpoint-$CKPT_NUM \
--output_dir=$OUT_DIR/$CKPT_NUM \
--test_qa_path=$DATA_DIR/trivia-test.qa.csv \
--train_qa_path=$DATA_DIR/trivia-train.qa.csv \
--dev_qa_path=$DATA_DIR/trivia-dev.qa.csv \
--max_seq_length=256 --per_gpu_eval_batch_size=1024 \
--passage_path=$DATA_DIR/psgs_w100.tsv --fp16
```

### 3.  reformat top_k results to train cross-encoder ranker

```bash
EXP_NAME=run_de_ict_triviaqa
DATA_DIR=/mnt/data/denseIR/data/trivia_data/
CKPT_NUM=40000
OUT_DIR=../output/$EXP_NAME
TOPK_FILE=$OUT_DIR/$CKPT_NUM/dev_result_dict_list.json # dev_result_dict_list.json is generate in previous step
CE_TRAIN_FILE=$OUT_DIR/$CKPT_NUM/dev_ce_0_triviaqa.json
LABELED_FILE=$OUT_DIR/$CKPT_NUM/biencoder-trivia-dev.json

python ./utils/prepare_ce_data.py $TOPK_FILE $CE_TRAIN_FILE $LABELED_FILE   # generate dev set file


TOPK_FILE=$OUT_DIR/$CKPT_NUM/train_result_dict_list.json # train_result_dict_list.json is generate in previous step
CE_TRAIN_FILE=$OUT_DIR/$CKPT_NUM/train_ce_0_triviaqa.json
LABELED_FILE=$OUT_DIR/$CKPT_NUM/biencoder-trivia-train.json

python ./utils/prepare_ce_data.py $TOPK_FILE $CE_TRAIN_FILE $LABELED_FILE # generate train set file
```

### 4. Warm ranker

```bash
EXP_NAME=run_ce_model
OUT_DIR=../output/$EXP_NAME
DE_EXP_NAME=run_de_ict_triviaqa
DE_OUT_DIR=../output/$DE_EXP_NAME
DE_CKPT_NUM=40000
TB_DIR=../tensorboard_log/$EXP_NAME    # tensorboard log path

python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=9538 \
./wiki/run_ce_model_ernie.py \
--model_type=nghuyong/ernie-2.0-large-en --max_seq_length=256 \
--per_gpu_train_batch_size=1 --gradient_accumulation_steps=8 \
--number_neg=15 --learning_rate=1e-5 \
--output_dir=$OUT_DIR \
--origin_data_dir=$DE_OUT_DIR/$DE_CKPT_NUM/train_ce_0_triviaqa.json \
--origin_data_dir_dev=$DE_OUT_DIR/$DE_CKPT_NUM/dev_ce_0_triviaqa.json \
--warmup_steps=1000 --logging_steps=100 --save_steps=1000 \
--max_steps=10000 --log_dir=$TB_DIR \
--fp16
```

### 5. AR2 Training

```bash
EXP_NAME=co_training_triviaqa
TB_DIR=../tensorboard_log/$EXP_NAME    # tensorboard log path
OUT_DIR=../output/$EXP_NAME

DE_EXP_NAME=run_de_ict_triviaqa
CE_EXP_NAME=run_ce_model
DE_CKPT_PATH=../output/$DE_EXP_NAME/checkpoint-40000
CE_CKPT_PATH=../output/$CE_EXP_NAME/checkpoint-4000
Origin_Data_Dir=../output/$DE_EXP_NAME/40000/train_ce_0_triviaqa.json
Origin_Data_Dir_Dev=../output/$DE_EXP_NAME/40000/dev_ce_0_triviaqa.json

Reranker_TYPE=nghuyong/ernie-2.0-large-en
Iteration_step=2000 
Iteration_reranker_step=500
MAX_STEPS=32000

# for global_step in `seq 0 2000 $MAX_STEPS`; do echo $global_step; done;
for global_step in `seq 0 $Iteration_step $MAX_STEPS`; 
do 
    python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=9539 $BASE_SCRIPT_DIR/wiki/co_training_wiki_train.py \
    --model_type=nghuyong/ernie-2.0-en \
    --model_name_or_path=$DE_CKPT_PATH \
    --max_seq_length=128 --per_gpu_train_batch_size=8 --gradient_accumulation_steps=1 \
    --number_neg=15 --learning_rate=1e-5 \
    --reranker_model_type=$Reranker_TYPE \
    --reranker_model_path=$CE_CKPT_PATH \
    --reranker_learning_rate=1e-6 \
    --output_dir=$OUT_DIR \
    --log_dir=$TB_DIR \
    --origin_data_dir=$Origin_Data_Dir \
    --origin_data_dir_dev=$Origin_Data_Dir_Dev \
    --warmup_steps=2000 --logging_steps=10 --save_steps=2000 --max_steps=$MAX_STEPS \
    --gradient_checkpointing --normal_loss \
    --iteration_step=$Iteration_step \
    --iteration_reranker_step=$Iteration_reranker_step \
    --temperature_normal=1 --ann_dir=$OUT_DIR/temp --adv_lambda 0.5 --global_step=$global_step

    g_global_step=`expr $global_step + $Iteration_step`
    python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=9539 $BASE_SCRIPT_DIR/wiki/co_training_wiki_generate.py \
    --model_type=nghuyong/ernie-2.0-en \
    --model_name_or_path=$Warm_de_path \
    --max_seq_length=128 --per_gpu_train_batch_size=8 --gradient_accumulation_steps=1 \
    --number_neg=15 --learning_rate=1e-5 \
    --reranker_model_type=$Reranker_TYPE \
    --reranker_model_path=$Warm_Reranker_PATH \
    --reranker_learning_rate=1e-6 \
    --output_dir=$BASE_DIR/ckpt/$EXP_NAME \
    --log_dir=tensorboard/logs/$EXP_NAME \
    --origin_data_dir=$BASE_DIR/ckpt/run_de_model_ict_ernie_triviaqa/40k/train_ce_0_triviaqa.json \
    --origin_data_dir_dev=$BASE_DIR/ckpt/run_de_model_ict_ernie_triviaqa/40k/dev_ce_0_triviaqa.json \
    --train_qa_path=$BASE_DIR/data/trivia_data/trivia-train.qa.csv \
    --test_qa_path=$BASE_DIR/data/trivia_data/trivia-test.qa.csv \
    --dev_qa_path=$BASE_DIR/data/trivia_data/trivia-dev.qa.csv \
    --passage_path=$BASE_DIR/data/psgs_w100.tsv \
    --warmup_steps=2000 --logging_steps=10 --save_steps=2000 --max_steps=$MAX_STEPS \
    --gradient_checkpointing --normal_loss --adv_step=0 \
    --iteration_step=$Iteration_step \
    --iteration_reranker_step=$Iteration_reranker_step \
    --temperature_normal=1 --ann_dir=$BASE_DIR/ckpt/$EXP_NAME/temp --adv_lambda=0.5 --global_step=$g_global_step
done
```

```

```
