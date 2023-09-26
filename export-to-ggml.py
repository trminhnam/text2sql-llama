############ Reference ############
# https://discuss.huggingface.co/t/help-with-merging-lora-weights-back-into-base-model/40968/4
###################################


import os
import subprocess

from transformers import HfArgumentParser

from utils.load_model import load_model_with_peft_and_tokenizer

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class QuantizeArguments:
    model_name_or_path: Optional[str] = field(default=None)
    model_name_or_path_subfolder: Optional[str] = field(default="")

    # TODO: LoRA parameters
    peft_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name or path of the pretrained LoRA model to use for the adapter."
        },
    )
    peft_name_or_path_subfolder: str = field(
        default="",
        metadata={
            "help": "The subfolder of the pretrained LoRA model to use for the adapter."
        },
    )

    llama_cpp_dir: Optional[str] = field(
        default="./llama.cpp",
        metadata={"help": "The directory where the llama.cpp repo is located."},
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )

    quant_mode: str = field(
        default="",
        metadata={
            "help": "The quantization mode to use for the model. Options are `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q_2`, `Q3_K`, `Q3_K_S`, `Q3_K_M`, `Q3_K_L`, `Q4_K`, `Q4_K_S`, `Q4_K_M`, `Q5_K`, `Q5_K_S`, `Q5_K_M`, `Q6_K`, `Q8_0`, `F16`, `F32`, `COPY`."
        },
    )

    output_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "The output directory where the quantized model and tokenizer will be written."
        },
    )

    def __str__(self):
        self_as_dict = asdict(self)

        self_as_dict = {
            k: f"<{k.upper()}>" if k.endswith("_token") else v
            for k, v in self_as_dict.items()
        }
        return f"{self.__class__.__name__}" + json.dumps(self_as_dict, indent=2) + "\n"

        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"


if __name__ == "__main__":
    print("#" * 80)
    print("Loading quantize args...")
    parser = HfArgumentParser((QuantizeArguments,))
    quantize_args = parser.parse_args_into_dataclasses()[0]
    print(f"Quantize args: {quantize_args}")

    model, tokenizer = load_model_with_peft_and_tokenizer(quantize_args, {})
    print(f"Model:\n{model}")
    print("#" * 80)
    print()

    save_dirs = [quantize_args.output_dir]
    if quantize_args.peft_name_or_path is not None:
        save_dirs.append(quantize_args.peft_name_or_path)
    if quantize_args.peft_name_or_path_subfolder is not None:
        save_dirs.append(quantize_args.peft_name_or_path_subfolder)
    save_dir = os.path.join(*save_dirs)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Saving tokenizer to {save_dir}")
    tokenizer.save_pretrained(save_dir)
    print("#" * 80)
    print()

    if quantize_args.peft_name_or_path is not None:
        model = model.merge_and_unload()
        print(f"Merged Model:\n{model}")
        print("#" * 80)
        print()

    # convert to fp16 and save
    model = model.half()
    model.save_pretrained(save_dir)
    print(f"Saved FP16 model to {save_dir}")
    print("#" * 80)
    print()

    # save quantized model
    print(f"Converting to GGML...")
    ggml_path = os.path.join(save_dir, "ggml-model-f16.gguf")
    if not os.path.exists(ggml_path):
        convert_commands = [
            "python",
            os.path.join(quantize_args.llama_cpp_dir, "convert.py"),
            save_dir,
            "--outtype",
            "f16",
        ]
        subprocess.call(convert_commands)
        print(f"Saved GGML model to {ggml_path}")
    else:
        print(f"GGML model already exists at {ggml_path}")
    print("#" * 80)
    print()

    # quantize the model to k-bit with quant_mode
    print(f"Quantizing to {quantize_args.quant_mode}...")
    quantized_path = os.path.join(
        save_dir, f"ggml-model-{quantize_args.quant_mode.lower()}.gguf"
    )
    if not os.path.exists(quantized_path):
        quantize_commands = [
            os.path.join(quantize_args.llama_cpp_dir, "build", "bin", "quantize"),
            ggml_path,
            quantized_path,
            quantize_args.quant_mode,
        ]
        subprocess.call(quantize_commands)
        print(f"Saved quantized model to {quantized_path}")
    else:
        print(f"Quantized model already exists at {quantized_path}")
    print("#" * 80)
    print()

    print("Done!")
