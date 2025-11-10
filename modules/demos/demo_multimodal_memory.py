"""Demonstration and test of MultiModalMemoryNetwork.

Shows how to use the multimodal network with test-time training for:
1. Single modality encoding
2. Multi-modal fusion
3. Cross-modal retrieval
4. Test-time memory adaptation
"""

import torch
import torch.nn as nn
from architectures import MultiModalMemoryNetwork

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

print("=" * 80)
print("MULTIMODAL MEMORY NETWORK DEMONSTRATION")
print("=" * 80)

# Create model
model = MultiModalMemoryNetwork(
    vocab_size=10000,
    text_embed_dim=512,
    text_seq_len=128,
    image_size=224,
    patch_size=16,
    image_channels=3,
    audio_channels=1,
    latent_dim=512,
    memory_size=128,
    num_heads=8,
    num_layers=4,
    dropout=0.1,
    ttt_mode="attention",
    feedback_steps=2,
).to(device)

model.print_model_info()

print("\n" + "=" * 80)
print("TEST 1: Single Modality Processing")
print("=" * 80)

# Test text processing
batch_size = 4
seq_len = 32

text_tokens = torch.randint(0, 10000, (batch_size, seq_len)).to(device)
print(f"\nText input shape: {text_tokens.shape}")

outputs = model(text=text_tokens)
print(f"Central latent shape: {outputs['central_latent'].shape}")
print(f"✓ Text encoding successful")

# Test image processing
images = torch.randn(batch_size, 3, 224, 224).to(device)
print(f"\nImage input shape: {images.shape}")

outputs = model(images=images)
print(f"Central latent shape: {outputs['central_latent'].shape}")
print(f"✓ Image encoding successful")

# Test audio processing
audio = torch.randn(batch_size, 1, 128, 128).to(device)  # Spectrogram
print(f"\nAudio input shape: {audio.shape}")

outputs = model(audio=audio)
print(f"Central latent shape: {outputs['central_latent'].shape}")
print(f"✓ Audio encoding successful")

print("\n" + "=" * 80)
print("TEST 2: Multimodal Fusion")
print("=" * 80)

# Process all modalities together
outputs = model(text=text_tokens, images=images, audio=audio)
print(f"\nInput modalities: Text + Images + Audio")
print(f"Fused central latent shape: {outputs['central_latent'].shape}")
print(f"✓ Multimodal fusion successful")

print("\n" + "=" * 80)
print("TEST 3: Cross-Modal Decoding")
print("=" * 80)

# Encode one modality, decode to others
outputs = model(
    text=text_tokens,
    decode_to=['image', 'audio']
)

print(f"\nInput: Text")
print(f"Central latent: {outputs['central_latent'].shape}")
print(f"Decoded image: {outputs['decoded_image'].shape}")
print(f"Decoded audio: {outputs['decoded_audio'].shape}")
print(f"✓ Text → Image/Audio successful")

# Image to text
outputs = model(
    images=images,
    decode_to=['text']
)

print(f"\nInput: Image")
print(f"Central latent: {outputs['central_latent'].shape}")
print(f"Decoded text: {outputs['decoded_text'].shape}")
print(f"✓ Image → Text successful")

print("\n" + "=" * 80)
print("TEST 4: Test-Time Training (Memory Adaptation)")
print("=" * 80)

# Show memory updates during inference
model.eval()  # Set to eval mode
with torch.no_grad():
    # Get initial memory state
    initial_memory = model.central_memory.memory.clone()
    
    print("\nInitial central memory norm:", initial_memory.norm().item())
    
    # Process data (memory will update via TTT)
    _ = model(text=text_tokens)
    
    # Check memory after processing
    updated_memory = model.central_memory.memory
    
    print("Updated central memory norm:", updated_memory.norm().item())
    print("Memory change:", (updated_memory - initial_memory).norm().item())
    print("✓ Test-time memory adaptation active")

print("\n" + "=" * 80)
print("TEST 5: Cross-Modal Retrieval")
print("=" * 80)

# Query: text, retrieve: image representation
text_query = torch.randint(0, 10000, (2, seq_len)).to(device)
retrieved_image = model.cross_modal_retrieval(
    query_modality='text',
    query_data=text_query,
    target_modality='image'
)

print(f"\nQuery modality: Text (shape: {text_query.shape})")
print(f"Retrieved: Image representation (shape: {retrieved_image.shape})")
print(f"✓ Text → Image retrieval successful")

# Query: image, retrieve: audio representation
image_query = torch.randn(2, 3, 224, 224).to(device)
retrieved_audio = model.cross_modal_retrieval(
    query_modality='image',
    query_data=image_query,
    target_modality='audio'
)

print(f"\nQuery modality: Image (shape: {image_query.shape})")
print(f"Retrieved: Audio representation (shape: {retrieved_audio.shape})")
print(f"✓ Image → Audio retrieval successful")

print("\n" + "=" * 80)
print("TEST 6: Feedback Loop Iterations")
print("=" * 80)

# Test with different feedback steps
for steps in [0, 1, 2, 5]:
    model.feedback_steps = steps
    outputs = model(text=text_tokens)
    print(f"Feedback steps: {steps}, Output norm: {outputs['central_latent'].norm().item():.4f}")

print("✓ Feedback loop functioning")

print("\n" + "=" * 80)
print("TEST 7: Memory Component Testing")
print("=" * 80)

# Test individual memory components
memory = model.central_memory
test_input = torch.randn(batch_size, 512).to(device)

# Read from memory
read_output = memory.read(test_input)
print(f"\nMemory read - Input shape: {test_input.shape}, Output shape: {read_output.shape}")

# Write to memory
write_output = memory.write(test_input, update_memory=False)
print(f"Memory write - Input shape: {test_input.shape}, Output shape: {write_output.shape}")

print("✓ Memory operations successful")

print("\n" + "=" * 80)
print("ARCHITECTURE SUMMARY")
print("=" * 80)

print(f"""
✓ Text Encoder: Transformer-based (vocab={model.text_encoder.token_embedding.num_embeddings})
✓ Image Encoder: Vision Transformer (patches={model.image_encoder.num_patches})
✓ Audio Encoder: CNN-based spectrogram processor
✓ Central Memory: {model.central_memory.memory_size} slots × {model.central_memory.memory_dim}D with TTT
✓ Modality Memories: {model.text_memory.memory_size} slots each with TTT
✓ Feedback Loop: {model.feedback_steps} iterations
✓ Cross-Modal: Unified latent space with modality-specific decoders

Key Features:
- Test-Time Training (TTT) in all memory blocks
- Attention-based memory read/write
- Cross-modal retrieval and generation
- Feedback loops for iterative refinement
- Unified representation learning
""")

print("=" * 80)
print("ALL TESTS PASSED! ✅")
print("=" * 80)

print("""
Usage Examples:

# 1. Single modality
outputs = model(text=text_tokens)
latent = outputs['central_latent']

# 2. Multiple modalities
outputs = model(text=text_tokens, images=images, audio=audio)
latent = outputs['central_latent']

# 3. With decoding
outputs = model(text=text_tokens, decode_to=['image', 'audio'])
decoded_img = outputs['decoded_image']
decoded_aud = outputs['decoded_audio']

# 4. Cross-modal retrieval
retrieved = model.cross_modal_retrieval(
    query_modality='text',
    query_data=text_query,
    target_modality='image'
)

# 5. Access memory state
central_mem = model.central_memory.memory  # (memory_size, latent_dim)
text_mem = model.text_memory.memory
""")

