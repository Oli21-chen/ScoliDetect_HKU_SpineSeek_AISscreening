from typing import Callable, Dict, List, Optional, Tuple

import math

import time



import torch

from torch import nn

from torch import amp

from torch.cuda.amp import GradScaler

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm





def _extract_text(batch: Dict) -> List[str]:

    prompts = batch.get("prompts", [])

    source_files = batch.get("source_files", [])

    texts: List[str] = []

    for idx, prompt_list in enumerate(prompts):

        if isinstance(prompt_list, list) and len(prompt_list) > 0:

            texts.append(". ".join(prompt_list))

        elif isinstance(prompt_list, str) and prompt_list:

            texts.append(prompt_list)

        else:

            fallback = source_files[idx] if idx < len(source_files) else ""

            texts.append(fallback)

    return texts





def train_one_epoch(

    model: nn.Module,

    dataloader: torch.utils.data.DataLoader,

    optimizer: torch.optim.Optimizer,

    loss_fn: Callable[..., torch.Tensor],

    device: torch.device,

    scaler: Optional[GradScaler] = None,

    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,

    use_amp: bool = True,

    epoch: int = 0,

    global_step: int = 0,

    verbose: bool = True,

    gradient_accumulation_steps: int = 1,

    writer: Optional[SummaryWriter] = None,

    max_grad_norm: Optional[float] = None,

) -> Tuple[float, int]:

    """

    Train for one epoch with detailed step-by-step metrics.



    Returns:

        tuple: (average_loss, updated_global_step)

    """

    model.train()

    total_loss = 0.0

    valid_steps = 0



    if verbose:

        print(f"\n{'='*80}")

        print(f"Epoch {epoch} - Training")

        print(f"{'='*80}")



    progress = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False, disable=verbose)

    optimizer.zero_grad(set_to_none=True)

    current_lr = optimizer.param_groups[0]["lr"]



    for batch_idx, batch in enumerate(progress):

        video = batch["video"].to(device)

        km = batch["knowledge_map"].to(device)

        texts = _extract_text(batch)



        t_step_start = time.perf_counter()



        if use_amp and scaler is not None:

            with amp.autocast(device_type="cuda"):

                outputs = model(video, km, texts)

            with amp.autocast(device_type="cuda", enabled=False):

                result = loss_fn(outputs)

                loss, loss_details = (

                    (result[0], result[1]) if isinstance(result, tuple) else (result, None)

                )

                loss = loss / gradient_accumulation_steps

        else:

            outputs = model(video, km, texts)

            result = loss_fn(outputs)

            loss, loss_details = (

                (result[0], result[1]) if isinstance(result, tuple) else (result, None)

            )

            loss = loss / gradient_accumulation_steps



        step_loss_val = float(loss.detach().item() * gradient_accumulation_steps)

        if not math.isfinite(step_loss_val):

            print(

                f"⚠️  Epoch {epoch} batch {batch_idx + 1}: non-finite loss={step_loss_val}, skipping step"

            )

            optimizer.zero_grad(set_to_none=True)

            del outputs

            continue



        if use_amp and scaler is not None:

            scaler.scale(loss).backward()

        else:

            loss.backward()



        del outputs



        is_accum_step = (

            (batch_idx + 1) % gradient_accumulation_steps == 0

            or (batch_idx + 1) == len(dataloader)

        )

        if is_accum_step:

            if use_amp and scaler is not None:

                if max_grad_norm is not None:

                    scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                scaler.step(optimizer)

                scaler.update()

            else:

                if max_grad_norm is not None:

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()

            optimizer.zero_grad(set_to_none=True)



            if scheduler is not None:

                scheduler.step()



            if scheduler is not None:

                current_lr = scheduler.get_last_lr()[0]

            else:

                current_lr = optimizer.param_groups[0]["lr"]



            global_step += 1



            if writer is not None:

                step_loss = loss.item() * gradient_accumulation_steps

                writer.add_scalar("Loss/train_step", step_loss, global_step)

                writer.add_scalar("LR/step", current_lr, global_step)

                if loss_details is not None:

                    writer.add_scalar("Loss/train_km_text", loss_details["loss_km_text"], global_step)

                    writer.add_scalar(

                        "Loss/train_video_text", loss_details["loss_video_text"], global_step

                    )

                    writer.add_scalar("Loss/train_video_km", loss_details["loss_video_km"], global_step)



        total_loss += step_loss_val

        valid_steps += 1

        step_time = time.perf_counter() - t_step_start



        if verbose:

            avg_loss = total_loss / max(valid_steps, 1)

            accum_info = (

                f" [{batch_idx % gradient_accumulation_steps + 1}/{gradient_accumulation_steps}]"

                if gradient_accumulation_steps > 1

                else ""

            )

            if scheduler is not None:

                current_lr = scheduler.get_last_lr()[0]

            comp_info = ""

            if loss_details is not None:

                comp_info = (

                    f" | km-t: {loss_details['loss_km_text']:.4f} "

                    f"v-t: {loss_details['loss_video_text']:.4f} "

                    f"v-k: {loss_details['loss_video_km']:.4f}"

                )

            print(

                f"Epoch {epoch:3d} | Step {global_step:5d} | "

                f"Batch {batch_idx+1:3d}/{len(dataloader):3d}{accum_info} | "

                f"Loss: {step_loss_val:.6f} | Avg: {avg_loss:.6f} | "

                f"LR: {current_lr:.6e}{comp_info} | Time: {step_time:.3f} s"

            )

        else:

            progress.set_postfix(loss=step_loss_val, avg_loss=total_loss / max(valid_steps, 1))



    progress.close()

    return total_loss / max(valid_steps, 1), global_step





@torch.no_grad()

def evaluate(

    model: nn.Module,

    dataloader: torch.utils.data.DataLoader,

    loss_fn: Callable[..., torch.Tensor],

    device: torch.device,

    writer: Optional[SummaryWriter] = None,

    global_step: int = 0,

) -> Tuple[float, Optional[Dict[str, float]]]:

    """

    Returns:

        (average_loss, loss_details or None). loss_details has keys loss_km_text, loss_video_text, loss_video_km.

    """

    model.eval()

    total_loss = 0.0

    steps = 0

    sum_km_text = 0.0

    sum_video_text = 0.0

    sum_video_km = 0.0

    has_details = False



    progress = tqdm(dataloader, desc="Eval", leave=True)

    for batch in progress:

        video = batch["video"].to(device)

        km = batch["knowledge_map"].to(device)

        texts = _extract_text(batch)



        outputs = model(video, km, texts)

        result = loss_fn(outputs)

        loss, loss_details = (result[0], result[1]) if isinstance(result, tuple) else (result, None)

        total_loss += loss.item()

        steps += 1

        if loss_details is not None:

            sum_km_text += loss_details["loss_km_text"]

            sum_video_text += loss_details["loss_video_text"]

            sum_video_km += loss_details["loss_video_km"]

            has_details = True

        progress.set_postfix(loss=loss.item())



    progress.close()

    avg_loss = total_loss / max(steps, 1)

    details = None

    if has_details and steps > 0:

        details = {

            "loss_km_text": sum_km_text / steps,

            "loss_video_text": sum_video_text / steps,

            "loss_video_km": sum_video_km / steps,

        }

        if writer is not None:

            writer.add_scalar("Loss/val_km_text", details["loss_km_text"], global_step)

            writer.add_scalar("Loss/val_video_text", details["loss_video_text"], global_step)

            writer.add_scalar("Loss/val_video_km", details["loss_video_km"], global_step)

    return avg_loss, details


