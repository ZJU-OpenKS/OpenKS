#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The training and interaction process for knowledge-grounded dialogue generation"""

import sys
sys.path.append("/abs/path/to/OpenKS/")
# or export PYTHONPATH="/abs/path/to/OpenKS/":$PYTHONPATH
import argparse
from collections import namedtuple
import json

from termcolor import colored, cprint
import paddle.fluid as fluid

import openks.apps.dialog.models as models
import openks.apps.dialog.tasks as tasks
from openks.apps.dialog.tasks.dialog_generation import DialogGeneration
from openks.apps.dialog.utils import check_cuda, Timer
from openks.apps.dialog.utils.args import parse_args, str2bool


def load_conf(args_path):
    """
    load train or interaction config
    """
    server_args = []
    with open(args_path) as fp:
        for line in fp:
            arg = line.strip()
            if arg == "" or arg.startswith("#"):
                continue
            server_args.append("--" + arg)
    return server_args


def setup_args(phase):
    """
    setup interaction or train arguments.
    """
    parser = argparse.ArgumentParser()
    if phase == "train":
        parser.add_argument("--is_distributed", type=str2bool, default=False)
        parser.add_argument("--save_path", type=str, default="./openks/apps/dialog/output")
        parser.add_argument("--train_file", type=str, default="./openks/apps/dialog/data/train_filelist")
        parser.add_argument("--valid_file", type=str, default="./openks/apps/dialog/data/valid_filelist")

        parser.add_argument("--start_step", type=int, default=0)
        parser.add_argument("--num_epochs", type=int, default=20)
        parser.add_argument("--log_steps", type=int, default=100)
        parser.add_argument("--validation_steps", type=int, default=1000)
        parser.add_argument("--save_steps", type=int, default=5000)
        models.add_cmdline_args(parser)
        tasks.add_cmdline_args(parser)
    elif phase == "interaction":
        models.add_cmdline_args(parser)
        DialogGeneration.add_cmdline_args(parser)

    if phase == "interaction":
        server_args = load_conf("./openks/apps/dialog/package/dialog_zh/interaction.conf")
    elif phase == "train":
        server_args = load_conf("./openks/apps/dialog/package/dialog_en/train.conf")
    args = parse_args(parser, input_args=server_args)
    args.load(args.config_path, "Model")
    
    if phase == "interaction":
        args.run_infer = True # only build infer program
    print(json.dumps(args, indent=2))
    return args


def interact(args):
    """
    Interaction main function.
    """
    dev_count = 1
    gpu_id = 0
    place = fluid.CUDAPlace(gpu_id)

    task = DialogGeneration(args)
    model = models.create_model(args, place)

    Example = namedtuple("Example", ["src", "data_id"])
    context = []
    start_info = "Enter [EXIT] to quit the interaction, [NEXT] to start a new conversation."
    cprint(start_info, "yellow", attrs=["bold"])
    while True:
        user_utt = input(colored("[Human]: ", "red", attrs=["bold"])).strip()
        if user_utt == "[EXIT]":
            break
        elif user_utt == "[NEXT]":
            context = []
            cprint(start_info, "yellow", attrs=["bold"])
        else:
            context.append(user_utt)
            example = Example(src=" [SEP] ".join(context), data_id=0)
            record = task.reader._convert_example_to_record(example, is_infer=True)
            data = task.reader._pad_batch_records([record], is_infer=True)
            pred = task.infer_step(model, data)[0]
            bot_response = pred["response"]
            print(colored("[Bot]:", "blue", attrs=["bold"]), colored(bot_response, attrs=["bold"]))
            context.append(bot_response)

    return


def train(args):
    """
    Train main function.
    """
    if args.is_distributed:
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)

        dev_count = fluid.core.get_cuda_device_count()
        gpu_id = int(os.getenv("FLAGS_selected_gpus"))
        trainers_num = fleet.worker_num()
        trainer_id = fleet.worker_index()
    else:
        dev_count = 1
        gpu_id = 0
        trainers_num = 1
        trainer_id = 0
    place = fluid.CUDAPlace(gpu_id)

    task = tasks.create_task(args)
    model = models.create_model(args, place)
    train_generator = task.get_data_loader(
        model,
        input_file=args.train_file,
        num_epochs=args.num_epochs,
        num_part=trainers_num,
        part_id=trainer_id,
        phase="train"
    )
    valid_generator = task.get_data_loader(
        model,
        input_file=args.valid_file,
        num_part=dev_count,
        part_id=gpu_id,
        phase="distributed_valid" if args.is_distributed else "valid"
    )

    # run training
    timer = Timer()
    timer.start()
    for step, data in enumerate(train_generator(), args.start_step + 1):
        outputs = task.train_step(model, data)
        timer.pause()
        if step % args.log_steps == 0:
            time_cost = timer.pass_time
            current_epoch, current_file_index, total_file = task.reader.get_train_progress()
            print(f"[train][{current_epoch}] progress: {current_file_index}/{total_file} "
                  f"step: {step}, time: {time_cost:.3f}, "
                  f"speed: {args.log_steps / time_cost:.3f} steps/s")
            print(f"\tcurrent lr: {outputs.pop('scheduled_lr'):.7f}")
            metrics = task.get_metrics(outputs)
            print("\t" + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
            timer.reset()

        if step % args.validation_steps == 0:
            evaluate(task, model, valid_generator, args, dev_count, gpu_id, step)

        if step % args.save_steps == 0 and trainer_id == 0:
            save_path = f"{args.save_path}/step_{step}"
            model.save(save_path, is_checkpoint=True)
            with open(save_path + ".finish", "w") as f:
                pass

        timer.start()


def evaluate(task, model, generator, args, dev_count, gpu_id, training_step):
    """
    evaluate
    """
    outputs = None
    print("=" * 80)
    print("Evaluation:")
    timer = Timer()
    timer.start()
    for step, data in enumerate(generator(), 1):
        part_outputs = task.eval_step(model, data)
        outputs = task.merge_mertrics_and_statistics(outputs, part_outputs)

        if step % args.log_steps == 0:
            metrics = task.get_metrics(outputs)
            print(f"\tstep {step}:" + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

    if args.is_distributed:
        # merge evaluation outputs in distributed mode.
        part_file = os.path.join(args.save_path, f"evaluation_output.part_{gpu_id}")
        with open(part_file, "w") as fp:
            json.dump(outputs, fp, ensure_ascii=False)
        part_finish_file = os.path.join(args.save_path, f"evaluation_output.part_{gpu_id}.finish")
        with open(part_finish_file, "w"):
            pass

        if gpu_id == 0:
            part_files = f"evaluation_output.part_*.finish"
            while True:
                ret = subprocess.getoutput(f"find {args.save_path} -maxdepth 1 -name {part_files}")
                num_completed = len(ret.split("\n"))
                if num_completed != dev_count:
                    time.sleep(1)
                    continue
                outputs = None
                for dev_id in range(dev_count):
                    part_file = os.path.join(args.save_path, f"evaluation_output.part_{dev_id}")
                    with open(part_file, "r") as fp:
                        part_outputs = json.load(fp)
                        outputs = task.merge_mertrics_and_statistics(outputs, part_outputs)
                break
            subprocess.getoutput("rm " + os.path.join(args.save_path, f"evaluation_output.part*"))

    if gpu_id == 0:
        metrics = task.get_metrics(outputs)
        print(f"[Evaluation][{training_step}]" + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
    print(f"\ttime cost: {timer.pass_time:.3f}")
    print("=" * 80)
    return


if __name__ == "__main__":
    ## Usage: python -m examples.dialog_generation_example phase[train|interaction]
    ## Run Environments: CUDA version=9.2, python >= 3.7.0, paddlepaddle >= 1.8.2
    phase = sys.argv[1]
    args = setup_args(phase)
    check_cuda(True)
    if phase == "interaction":
        interact(args)
    elif phase == "train":
        train(args)
    else:
        raise Exception("The value for phase is only one of ['train', 'interaction']")
