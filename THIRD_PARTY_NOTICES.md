# Third-party attribution review

Reviewed for the `0.11` release line on 2026-09-08. MOLT's wheel contains MOLT
Python source and legal files; it does not vendor the libraries below. Package
installers resolve them as separate distributions governed by their own terms.

## Direct runtime dependencies

| Package | MOLT constraint | Declared license | Upstream |
| --- | --- | --- | --- |
| NumPy | `>=2.1,<3` | BSD-3-Clause and bundled component licenses | [numpy/numpy](https://github.com/numpy/numpy) |
| nvidia-ml-py | `>=13.580.65,<14` | BSD | [PyPI project](https://pypi.org/project/nvidia-ml-py/) |
| psutil | `>=7,<8` | BSD-3-Clause | [giampaolo/psutil](https://github.com/giampaolo/psutil) |
| PyTorch | `>=2.8,<2.9` | BSD-3-Clause; upstream distribution includes a NOTICE | [pytorch/pytorch](https://github.com/pytorch/pytorch) |

## Optional runtime dependencies

| Extra | Package | MOLT constraint | Declared license | Upstream |
| --- | --- | --- | --- | --- |
| `qlora` | Accelerate | `>=1.10,<2` | Apache-2.0 | [huggingface/accelerate](https://github.com/huggingface/accelerate) |
| `qlora` | bitsandbytes | `>=0.48,<1` | MIT | [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) |
| `qlora` | PEFT | `>=0.17,<1` | Apache-2.0 | [huggingface/peft](https://github.com/huggingface/peft) |
| `qlora` | Transformers | `>=5,<6` | Apache-2.0 | [huggingface/transformers](https://github.com/huggingface/transformers) |
| `data` | PyArrow | `>=19,<24` | Apache-2.0; upstream distribution includes a NOTICE | [apache/arrow](https://github.com/apache/arrow) |
| `windows-fusion` | triton-windows | `>=3.4,<3.5` | MIT | [woct0rdho/triton-windows](https://github.com/woct0rdho/triton-windows) |

Development and build tools are not runtime components of the MOLT wheel.
Transitive dependencies are installed separately and retain their own notices.
Redistributors of a complete environment, appliance, or installer must review
the exact resolved dependency set and include any notices required by those
distributions. This record is an attribution inventory, not legal advice.
