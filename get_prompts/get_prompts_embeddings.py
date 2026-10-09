"""
Generate text embeddings for gait analysis prompts.

This script loads gait prompts from JSON and generates embeddings using various methods:
1. Sentence Transformers (default, recommended)
2. CLIP text encoder (for vision-language models)
3. OpenAI embeddings (API-based)

Output formats:
- NumPy arrays (.npy)
- PyTorch tensors (.pt)
- HDF5 files (.h5)
- JSON with metadata
"""

import json
import os
import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path


def load_prompts(json_path: str = "gait_prompts.json") -> Dict:
    """Load prompts from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_individual_prompts(json_path: str = "individual_prompts/individual_gait_prompts.json") -> List[str]:
    """
    Load and extract all prompts from individual gait prompts JSON.
    
    Args:
        json_path: Path to individual_gait_prompts.json file
    
    Returns:
        List of all unique prompts (forward, backward, both, top_feature_prompts)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_prompts = set()
    
    if 'subject_prompts' in data:
        for subject_id, subject_data in data['subject_prompts'].items():
            # Add forward prompts
            all_prompts.update(subject_data.get('forward', []))
            # Add backward prompts
            all_prompts.update(subject_data.get('backward', []))
            # Add both prompts
            all_prompts.update(subject_data.get('both', []))
            # Add top feature prompts
            all_prompts.update(subject_data.get('top_feature_prompts', []))
    
    return sorted(list(all_prompts))


def get_all_prompts(data: Dict, use_concise: bool = True) -> List[str]:
    """
    Extract all prompts from the loaded data.
    
    Args:
        data: Loaded JSON data
        use_concise: If True, use concise_prompts; else use all categorized prompts
    
    Returns:
        List of prompt strings
    """
    if use_concise:
        return data.get("concise_prompts", [])
    else:
        # Flatten all categorized prompts
        categorized = data.get("categorized_prompts", {})
        all_prompts = []
        for category, prompts in categorized.items():
            all_prompts.extend(prompts)
        return all_prompts


# ============================================================================
# Method 1: Sentence Transformers (Recommended)
# ============================================================================

def get_embeddings_sentence_transformers(
    prompts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
    batch_size: int = 32,
    show_progress: bool = True
) -> np.ndarray:
    """
    Generate embeddings using sentence-transformers.
    
    Args:
        prompts: List of text prompts
        model_name: Sentence transformer model name
                   Options: "all-MiniLM-L6-v2" (fast, 384-dim),
                           "all-mpnet-base-v2" (better, 768-dim),
                           "sentence-transformers/all-MiniLM-L6-v2"
        device: "cpu" or "cuda"
        batch_size: Batch size for encoding
        show_progress: Show progress bar
    
    Returns:
        NumPy array of shape (num_prompts, embedding_dim)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed. Install with: pip install sentence-transformers"
        )
    
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    
    print(f"Encoding {len(prompts)} prompts...")
    embeddings = model.encode(
        prompts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalize for cosine similarity
    )
    
    print(f"✅ Generated embeddings: shape {embeddings.shape}")
    return embeddings


# ============================================================================
# Method 2: CLIP Text Encoder
# ============================================================================

def get_embeddings_clip(
    prompts: List[str],
    model_name: str = "ViT-B/32",
    device: str = "cpu",
    batch_size: int = 32
) -> np.ndarray:
    """
    Generate embeddings using CLIP text encoder.
    
    Args:
        prompts: List of text prompts
        model_name: CLIP model name (e.g., "ViT-B/32", "ViT-L/14")
        device: "cpu" or "cuda"
        batch_size: Batch size for encoding
    
    Returns:
        NumPy array of shape (num_prompts, embedding_dim)
    """
    try:
        import clip
        import torch
    except ImportError:
        raise ImportError(
            "CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git"
        )
    
    print(f"Loading CLIP model: {model_name}")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    
    print(f"Encoding {len(prompts)} prompts...")
    embeddings_list = []
    
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            # Tokenize text
            text_tokens = clip.tokenize(batch, truncate=True).to(device)
            # Get embeddings
            text_features = model.encode_text(text_tokens)
            # Normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            embeddings_list.append(text_features.cpu().numpy())
    
    embeddings = np.vstack(embeddings_list)
    print(f"✅ Generated CLIP embeddings: shape {embeddings.shape}")
    return embeddings


# ============================================================================
# Method 3: OpenAI Embeddings (API)
# ============================================================================

def get_embeddings_openai(
    prompts: List[str],
    model: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    batch_size: int = 100
) -> np.ndarray:
    """
    Generate embeddings using OpenAI API.
    
    Args:
        prompts: List of text prompts
        model: OpenAI embedding model ("text-embedding-3-small", "text-embedding-3-large", etc.)
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        batch_size: Batch size for API calls
    
    Returns:
        NumPy array of shape (num_prompts, embedding_dim)
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package not installed. Install with: pip install openai"
        )
    
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter.")
    
    client = OpenAI(api_key=api_key)
    
    print(f"Encoding {len(prompts)} prompts using OpenAI {model}...")
    embeddings_list = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        response = client.embeddings.create(
            model=model,
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        embeddings_list.extend(batch_embeddings)
        print(f"  Processed {min(i + batch_size, len(prompts))}/{len(prompts)} prompts")
    
    embeddings = np.array(embeddings_list)
    print(f"✅ Generated OpenAI embeddings: shape {embeddings.shape}")
    return embeddings


# ============================================================================
# Save Functions
# ============================================================================

def save_embeddings(
    embeddings: np.ndarray,
    prompts: List[str],
    output_dir: str = "embeddings",
    method: str = "sentence_transformers",
    format: str = "npy"
):
    """
    Save embeddings in various formats.
    
    Args:
        embeddings: NumPy array of embeddings
        prompts: List of corresponding prompts
        output_dir: Output directory
        method: Embedding method name
        format: Output format ("npy", "pt", "h5", "json", or "all")
    """
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = f"gait_prompts_embeddings_{method}"
    
    if format == "all" or format == "npy":
        # NumPy format
        npy_path = os.path.join(output_dir, f"{base_name}.npy")
        np.save(npy_path, embeddings)
        print(f"💾 Saved NumPy array: {npy_path} (shape: {embeddings.shape})")
    
    if format == "all" or format == "pt":
        # PyTorch format
        try:
            import torch
            pt_path = os.path.join(output_dir, f"{base_name}.pt")
            torch.save(torch.from_numpy(embeddings), pt_path)
            print(f"💾 Saved PyTorch tensor: {pt_path}")
        except ImportError:
            print("⚠️  PyTorch not available, skipping .pt format")
    
    if format == "all" or format == "h5":
        # HDF5 format
        try:
            import h5py
            h5_path = os.path.join(output_dir, f"{base_name}.h5")
            with h5py.File(h5_path, 'w') as f:
                f.create_dataset('embeddings', data=embeddings)
                f.create_dataset('prompts', data=[p.encode('utf-8') for p in prompts])
                f.attrs['num_prompts'] = len(prompts)
                f.attrs['embedding_dim'] = embeddings.shape[1]
                f.attrs['method'] = method
            print(f"💾 Saved HDF5: {h5_path}")
        except ImportError:
            print("⚠️  h5py not available, skipping .h5 format")
    
    if format == "all" or format == "json":
        # JSON format with metadata
        json_path = os.path.join(output_dir, f"{base_name}.json")
        output_data = {
            "embeddings": embeddings.tolist(),
            "prompts": prompts,
            "metadata": {
                "num_prompts": len(prompts),
                "embedding_dim": int(embeddings.shape[1]),
                "method": method,
                "shape": list(embeddings.shape)
            }
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved JSON: {json_path}")
    
    # Also save prompts mapping
    mapping_path = os.path.join(output_dir, f"{base_name}_prompts.txt")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        for i, prompt in enumerate(prompts):
            f.write(f"{i}: {prompt}\n")
    print(f"💾 Saved prompt mapping: {mapping_path}")


# ============================================================================
# Main Function
# ============================================================================

def main(
    json_path: str = "gait_prompts.json",
    method: str = "sentence_transformers",
    model_name: Optional[str] = None,
    use_concise: bool = True,
    output_dir: str = "embeddings",
    output_format: str = "all",
    device: str = "cpu",
    use_individual: bool = False,
    prompts_list: Optional[List[str]] = None,
    **kwargs
):
    """
    Main function to generate and save embeddings.
    
    Args:
        json_path: Path to prompts JSON file
        method: Embedding method ("sentence_transformers", "clip", "openai")
        model_name: Model name (optional, uses defaults if None)
        use_concise: Use concise prompts (True) or all categorized prompts (False)
        output_dir: Output directory
        output_format: Output format ("npy", "pt", "h5", "json", or "all")
        device: Device for computation ("cpu" or "cuda")
        **kwargs: Additional arguments for specific methods
    """
    print("=" * 60)
    print("GAIT PROMPTS TEXT EMBEDDING GENERATION")
    print("=" * 60)
    
    # Load prompts
    if prompts_list is not None:
        # Use provided prompts list directly
        prompts = prompts_list
        print(f"\n📂 Using provided prompts list: {len(prompts)} prompts")
    elif use_individual:
        # Load individual prompts
        print(f"\n📂 Loading individual prompts from: {json_path}")
        prompts = load_individual_prompts(json_path)
        print(f"✅ Loaded {len(prompts)} individual prompts")
    else:
        # Load standard prompts
        print(f"\n📂 Loading prompts from: {json_path}")
        data = load_prompts(json_path)
        prompts = get_all_prompts(data, use_concise=use_concise)
        print(f"✅ Loaded {len(prompts)} prompts")
        
        if use_concise:
            print("   Using: concise_prompts")
        else:
            print("   Using: all categorized_prompts")
    
    # Generate embeddings
    print(f"\n🔮 Generating embeddings using: {method}")
    
    if method == "sentence_transformers":
        model_name = model_name or "all-MiniLM-L6-v2"
        embeddings = get_embeddings_sentence_transformers(
            prompts,
            model_name=model_name,
            device=device,
            **kwargs
        )
    elif method == "clip":
        model_name = model_name or "ViT-B/32"
        embeddings = get_embeddings_clip(
            prompts,
            model_name=model_name,
            device=device,
            **kwargs
        )
    elif method == "openai":
        model_name = model_name or "text-embedding-3-small"
        embeddings = get_embeddings_openai(
            prompts,
            model=model_name,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: sentence_transformers, clip, openai")
    
    # Save embeddings
    print(f"\n💾 Saving embeddings...")
    save_embeddings(
        embeddings,
        prompts,
        output_dir=output_dir,
        method=method,
        format=output_format
    )
    
    print("\n" + "=" * 60)
    print("✅ COMPLETE!")
    print("=" * 60)
    print(f"📊 Embeddings shape: {embeddings.shape}")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 60)
    
    return embeddings, prompts


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate text embeddings for gait prompts")
    parser.add_argument(
        "--json_path",
        type=str,
        default="gait_prompts.json",
        help="Path to prompts JSON file"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="sentence_transformers",
        choices=["sentence_transformers", "clip", "openai"],
        help="Embedding method"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name (uses default if not specified)"
    )
    parser.add_argument(
        "--use_concise",
        action="store_true",
        default=True,
        help="Use concise prompts (default: True)"
    )
    parser.add_argument(
        "--use_all",
        action="store_true",
        help="Use all categorized prompts instead of concise"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="embeddings",
        help="Output directory"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="all",
        choices=["npy", "pt", "h5", "json", "all"],
        help="Output format"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for computation"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--use_individual",
        action="store_true",
        help="Load individual prompts from individual_gait_prompts.json"
    )
    
    args = parser.parse_args()
    
    # Handle use_all flag
    use_concise = not args.use_all if args.use_all else args.use_concise
    
    main(
        json_path=args.json_path,
        method=args.method,
        model_name=args.model_name,
        use_concise=use_concise,
        output_dir=args.output_dir,
        output_format=args.output_format,
        device=args.device,
        use_individual=args.use_individual,
        batch_size=args.batch_size
    )
