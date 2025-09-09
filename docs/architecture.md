## JointDiT Model Architecture

**Overall Structure:** JointDiT combines a video diffusion UNet and an audio diffusion UNet into one model, adding transformer-based JointBlocks to allow the two modalities to interact. Conceptually, we have:

- **Video UNet Submodules:** V-Input, V-Expert(s), V-Output
- **Audio UNet Submodules:** A-Input, A-Expert(s), A-Output
- **JointDiT Blocks:** Transformer layers inserted between V-Expert and A-Expert layers (see Fig. 2 of the paper).

**Input Block:** The V-Input and A-Input take the noisy latent at time *t* (for video frames and audio spectrogram) and produce initial hidden representations:
- *In the paper:* these are the first layer of each pre-trained denoiser.
- *In code:* we flatten the spatial/temporal dimensions of the latents and linearly project the channel dimension to `d_model` (256 by default). This yields a sequence of tokens for video (length = T_v * S_v, e.g. frames×pixels) and for audio (length = T_a * S_a, e.g. time×frequency bins). This simplification replaces the actual conv layer but retains shape and information (each token corresponds to one spatial or spectral position of the input).

**Joint Block(s):** Each JointBlock is a modified transformer encoder block that operates on **both video and audio tokens**:
- **LayerNorms:** We maintain separate LN for video and audio paths (denoted LN_v and LN_a).
- **Cross-Modal Attention:** Using a Perceiver-style full attention, queries from video and audio attend to a combined set of keys from both modalities. This is implemented by concatenating the projected video and audio token sets (plus optional context tokens) inside `PerceiverJointAttention`. The attention is multi-headed (8 heads default) with a shared `d_model`, allowing each modality to influence the other at this layer.
- **Feed-Forward Networks:** After attention, the block has separate feed-forward MLPs for video and audio (FF_v and FF_a). These process each modality’s tokens independently (intra-modal refinement) before adding back to the token residual. Each FFN expands the dimension by `ff_mult` (2x by default) and uses a GEGLU activation.

Each JointBlock ends with updated video tokens and audio tokens of shape (batch, seq_length, d_model). Multiple JointBlocks can be stacked (the output of one feeding into the next). In a typical configuration, one JointBlock corresponds to one “Expert layer interaction” between the two UNets. For example, with 2 JointBlocks, we assume the original UNets had 2 intermediate layers where cross-attention occurs (see *Expert layers* in paper).

**Output Block:** After the final JointBlock, each modality’s tokens are transformed back to produce a prediction for the clean latent $x_0$ at that timestep:
- *In the paper:* the last layers of each UNet (all remaining layers after the Expert layers) act as the Output decoder.
- *In code:* we apply a sequence of `LazyAdaLN` adaptors (one per remaining original layer) to the tokens (these are simple learned normalizations meant to adjust channel statistics to what the pre-trained decoder would expect). Then we unflatten the token sequence back to the original spatial dimensions and apply a final linear projection to map from `d_model` (256) to the original channel count (video latent C_v=4, audio latent C_a=8). The result is a tensor of shape identical to the input latent, which is our predicted denoised latent $\hat{v}_0$ or $\hat{a}_0$. These are used to compute the denoising loss (L2 between predicted and true latent, or a variant depending on noise parameterization).

**AdaLN Adaptation:** We introduced **LazyAdaLN** layers for each original UNet layer to be integrated. These layers start as identity (no change), but are learned during training to compensate for differences between the pre-trained model’s distribution and the new joint model’s needs. For example, if the video UNet’s block 3 output needs scaling when audio is present, AdaLN can absorb that difference. We have 6 AdaLNs per modality by default (since our base UNets are 6-block U-Nets):
- AdaLN1 corresponds to the output of the first UNet block (Input layer output),
- AdaLN2 and AdaLN3 correspond to the outputs of the next two blocks (Expert layers 1 and 2),
- AdaLN4, AdaLN5, AdaLN6 correspond to the final three blocks (Output layers). In code, `forward_output` simply cascades AdaLN4→5→6.

**Use of Pre-trained Weights:** Both the video and audio UNets are initialized from pretrained diffusion models (e.g., StableVideoDiffusion for video, AudioLDM for audio). Initially, we keep these weights fixed. The joint training objective combines the video denoising loss and audio denoising loss (as in the paper’s unified loss). By freezing the original weights in Stage A, we ensure we don’t drift too far from the original models’ capability before the joint attention learns to sync them. The adaptive layers and JointBlocks learn to adjust the outputs gradually. In Stage B, we can then unfreeze some of these weights to fine-tune the models together on the I2SV task.

**Current Status / Cautions:** Our implementation is **functionally correct** in wiring the model, but for efficiency we did not run the internal conv layers of the UNets during joint training. Practically, this means the network’s capacity during training comes mostly from the transformer JointBlocks and the AdaLNs, **not** from deep conv feature processing. This could make training faster and avoid GPU Out-Of-Memory (OOM) issues, but it also **puts more burden on the transformer to learn generative features**. Users might notice that early results have lower detail; this is expected until fine-tuning (or a future update that incorporates the conv layers). We plan to experiment with enabling the conv layers (with grad off at first) to see if it improves fidelity without too much overhead.

In summary, JointDiT’s architecture in code is a **hybrid transformer+UNet**: a transformer for cross-modal interaction sitting in between slices of two UNets. It adheres to the published design on a high level, with some shortcuts in implementation geared toward practicality. These shortcuts (linear projections in, AdaLN in place of convs, etc.) are noted above, so you’re aware of where the model might be improved going forward.
