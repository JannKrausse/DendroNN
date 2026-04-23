import torch
import torch.nn.utils.prune as prune
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
import matplotlib
matplotlib.use('Agg')


def test_model_with_compression(model, test_loader, train_loader, val_loader, hyperparameters, logs_folder, folder_name, logger, 
                               acc, devices, pruning_level=0.9, bit_width=8, device=0):
    """
    Test a model both in original form and with compression applied.
    
    Args:
        model: The trained model to test
        test_loader: DataLoader for test data
        train_loader: DataLoader for training data (used for fine-tuning after pruning)
        hyperparameters: Dictionary of hyperparameters
        logs_folder: Base folder for logs
        folder_name: Folder name for this experiment
        acc: Accelerator type ("gpu" or "cpu")
        devices: Device list for training
        pruning_level: Fraction of weights to prune (default: 0.9)
        bit_width: Bit width for quantization (default: 8)
        device: GPU device index (default: 0)
    
    Returns:
        dict: Dictionary containing test results for both original and compressed models
    """
    from pl_snn import SpikingNetwork, ConfusionMatrixPlotterCallback
    
    results = {}
    
    # Test original model
    print("Testing original model...")
    original_trainer = Trainer(
        max_epochs=1,
        accelerator=acc,
        devices=devices,
        log_every_n_steps=1,
        enable_progress_bar=True,
        logger=logger,
    )
    original_results = original_trainer.test(model, test_loader)
    results['original'] = original_results[0] if original_results else {}
    print(f"Original model test accuracy: {results['original'].get('test_acc', 'N/A')}")
    
    # Test compressed model
    print("Testing compressed model...")
    
    # Create a copy of the model using state_dict instead of deepcopy
    compressed_model = SpikingNetwork(
        model.net,
        lr=hyperparameters["lr"],
        hyperparameters=hyperparameters,
        output_decoder=hyperparameters["output_decoder"],
        lr_scheduled=hyperparameters["lr_scheduled"],
        lr_update_freq=hyperparameters["lr_update_freq"],
        lr_decay=hyperparameters["lr_decay"],
        l2_regu=hyperparameters["l2_regu"],
    )
    compressed_model.load_state_dict(model.state_dict())
    compressed_model = compressed_model.to(f"cuda:{device}" if acc == "gpu" else "cpu")
    
    # Apply compression
    compressed_model = compress_model(compressed_model, train_loader, val_loader, pruning_level, bit_width, logger, device=device)
    
    # Analyze compression results
    compression_stats = analyze_compression(compressed_model)
    
    print(f"Compression targets vs actual:")
    print(f"  Target pruning: {pruning_level*100:.1f}%, Actual pruning: {compression_stats['sparsity_percentage']:.2f}%")
    print(f"  Target quantization: {bit_width} bits, Actual unique values: {compression_stats['num_unique_values']}")
    
    # Create separate logger for compressed model testing
    compression_logger = TensorBoardLogger(
        save_dir=logs_folder, 
        name=(folder_name + f"/compression_p{int(pruning_level * 100)}_q{bit_width}")
    )
    
    # Create separate callbacks for compression testing
    compression_callbacks = [
        ConfusionMatrixPlotterCallback(log_path=compression_logger.log_dir),
    ]
    compression_callbacks[0].quantized_model = True  # Mark as compressed model
    
    # Test compressed model
    test_trainer = Trainer(
        max_epochs=1,
        accelerator=acc,
        devices=devices,
        log_every_n_steps=1,
        enable_progress_bar=True,
        callbacks=compression_callbacks,
        logger=compression_logger,
    )
    compressed_results = test_trainer.test(compressed_model, test_loader)
    results['compressed'] = compressed_results[0] if compressed_results else {}
    results['compression_stats'] = compression_stats
    
    print(f"Compressed model test accuracy: {results['compressed'].get('test_acc', 'N/A')}")
    
    # Save compressed model checkpoint
    torch.save({
        'state_dict': compressed_model.state_dict(),
        'hyperparameters': hyperparameters,
        'compression_stats': {
            'pruning_level': pruning_level,
            'bit_width': bit_width,
            'actual_pruning_percentage': compression_stats['sparsity_percentage'],
            'num_unique_weights': compression_stats['num_unique_values'],
            'percentage_nonzero_weights': compression_stats['density_percentage'],
        }
    }, f"{compression_logger.log_dir}/compressed_model.ckpt")
    
    print(f"Compressed model saved to: {compression_logger.log_dir}/compressed_model.ckpt")
    
    return results


def analyze_compression(model, layer_name="out"):
    """Analyze the compression results of a model layer (assumes nn.Linear layer)."""
    layer = model.net.hidden_layer[layer_name]
    weights = layer.weight.data
    
    # Basic statistics
    total_weights = weights.numel()
    zero_weights = (weights == 0).sum().item()
    nonzero_weights = total_weights - zero_weights
    
    # Unique values
    unique_values = torch.unique(weights)
    num_unique = len(unique_values)
    
    # Weight range
    min_weight = weights.min().item()
    max_weight = weights.max().item()
    
    results = {
        'total_weights': total_weights,
        'zero_weights': zero_weights,
        'nonzero_weights': nonzero_weights,
        'sparsity_percentage': (zero_weights / total_weights) * 100,
        'density_percentage': (nonzero_weights / total_weights) * 100,
        'num_unique_values': num_unique,
        'min_weight': min_weight,
        'max_weight': max_weight,
        'unique_values': unique_values.cpu().numpy()
    }
    
    print(f"Compression Analysis for layer '{layer_name}':")
    print(f"  Total weights: {total_weights:,}")
    print(f"  Zero weights: {zero_weights:,} ({results['sparsity_percentage']:.2f}%)")
    print(f"  Non-zero weights: {nonzero_weights:,} ({results['density_percentage']:.2f}%)")
    print(f"  Unique values: {num_unique}")
    print(f"  Weight range: [{min_weight:.6f}, {max_weight:.6f}]")
    
    return results


DEVICE = 2


def quantize_weights_manual(weights, bit_width):
    """Manual integer quantization for signed weights (positive and negative values)"""
    if bit_width >= 16:  # No quantization needed for 16+ bits
        return weights
        
    # For signed quantization
    n_levels = 2 ** bit_width
    
    # Find the full range of the weights (handles both positive and negative)
    min_val = weights.min()
    max_val = weights.max()
    
    if torch.abs(min_val) < 1e-8 and torch.abs(max_val) < 1e-8:  # All weights are essentially zero
        return weights
    
    # Use asymmetric quantization to better utilize the available levels
    weight_range = max_val - min_val
    if weight_range == 0:
        return weights
    
    # Scale to use full quantization range [0, n_levels-1]
    scale = weight_range / (n_levels - 1)
    zero_point = torch.round(-min_val / scale)
    
    # Quantize: scale, shift, round, then reverse
    quantized_int = torch.round(weights / scale + zero_point)
    quantized_int = torch.clamp(quantized_int, 0, n_levels - 1)
    quantized = (quantized_int - zero_point) * scale
    
    # Alternative: Use symmetric quantization around zero for better stability
    # This is often better for neural networks
    abs_max = torch.max(torch.abs(min_val), torch.abs(max_val))
    scale_symmetric = (2 * abs_max) / (n_levels - 1)
    quantized_symmetric = torch.round(weights / scale_symmetric) * scale_symmetric
    quantized_symmetric = torch.clamp(quantized_symmetric, -abs_max, abs_max)
    
    # Choose the quantization with fewer unique values (more compressed)
    unique_asym = len(torch.unique(quantized))
    unique_sym = len(torch.unique(quantized_symmetric))
    
    if unique_sym <= unique_asym and unique_sym <= n_levels:
        final_quantized = quantized_symmetric
        method = "symmetric"
    else:
        final_quantized = quantized
        method = "asymmetric"

    # Verify quantization worked
    unique_vals = torch.unique(final_quantized)
    print(f"Quantization to {bit_width} bits ({method}): {len(unique_vals)} unique values (target ≤{n_levels})")
    print(f"L2 quantization error norm: {((final_quantized - weights)**2).sum().item()**0.5:.6f}")
    
    return final_quantized

class IterativePruningCallback(Callback):
    def __init__(self, 
                 target_pruning_level=0.9, 
                 pruning_step_size=0.05,
                 start_epoch=0,
                 prune_every_n_epochs=1,
                 layer_name="out",
                 adaptive_finetuning=True,
                 base_finetune_epochs=1):
        super().__init__()
        self.target_pruning_level = target_pruning_level
        self.pruning_step_size = pruning_step_size
        self.start_epoch = start_epoch
        self.prune_every_n_epochs = prune_every_n_epochs
        self.layer_name = layer_name
        self.current_pruning_total = 0.0
        self.pruning_steps_completed = 0
        self.adaptive_finetuning = adaptive_finetuning
        self.base_finetune_epochs = base_finetune_epochs
        
        # Calculate total steps needed
        self.total_steps = int(target_pruning_level / pruning_step_size)
        
    def get_adaptive_finetune_epochs(self):
        """Calculate how many epochs to finetune based on current pruning level."""
        if not self.adaptive_finetuning:
            return self.prune_every_n_epochs
        
        # Step-based thresholds
        if self.current_pruning_total < 0.6:
            multiplier = 1
        elif self.current_pruning_total < 0.8:
            multiplier = 2
        else:
            multiplier = 3
        
        return self.prune_every_n_epochs * multiplier
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Apply pruning at the start of specified epochs."""
        current_epoch = trainer.current_epoch
        
        # Calculate adaptive fine-tuning interval
        adaptive_interval = self.get_adaptive_finetune_epochs()
        
        # Check if we should prune this epoch
        if (current_epoch >= self.start_epoch and 
            (current_epoch - self.start_epoch) % adaptive_interval == 0 and
            self.current_pruning_total < self.target_pruning_level):
            
            # Increment step counter
            self.pruning_steps_completed += 1
            
            # Calculate target pruning for this step (incremental!)
            target_pruning_total = min(
                self.pruning_steps_completed * self.pruning_step_size,
                self.target_pruning_level
            )

            layer = pl_module.net.hidden_layer[self.layer_name]
            
            # Get current pruning level from existing mask (if any)
            if hasattr(layer, 'weight_mask'):
                zero_count = (layer.weight_mask == 0).sum().item()
                total_count = layer.weight_mask.numel()
                self.current_pruning_total = zero_count / total_count
            else:
                self.current_pruning_total = 0.0
            
            # Calculate amount to prune this step
            if self.current_pruning_total < 1.0:
                amount_to_prune = (target_pruning_total - self.current_pruning_total) / (1.0 - self.current_pruning_total)
            else:
                amount_to_prune = 0.0
            
            print(f"\n[Epoch {current_epoch}] Pruning step {self.pruning_steps_completed}/{self.total_steps}")
            print(f"  Current pruning: {self.current_pruning_total*100:.2f}%")
            print(f"  Target pruning: {target_pruning_total*100:.1f}%")
            print(f"  Amount to prune: {amount_to_prune*100:.2f}% of remaining weights")
            
            # Apply cumulative pruning by re-applying all pruning
            # PyTorch's pruning will combine masks automatically
            if amount_to_prune > 0:
                prune.l1_unstructured(layer, name="weight", amount=amount_to_prune)
                
                # Verify actual pruning level after this step
                if hasattr(layer, 'weight_mask'):
                    zero_count = (layer.weight_mask == 0).sum().item()
                    total_count = layer.weight_mask.numel()
                    actual_pruning = zero_count / total_count
                    print(f"  Actual pruning after step: {actual_pruning*100:.2f}%")
                    self.current_pruning_total = actual_pruning
            
            # Log pruning metrics to TensorBoard
            if trainer.logger is not None:
                trainer.logger.log_metrics({
                    'compression/current_pruning_level': self.current_pruning_total,
                    'compression/amount_to_prune': amount_to_prune,
                    'compression/pruning_step': self.pruning_steps_completed,
                }, step=trainer.global_step)

    def on_train_epoch_end(self, trainer, pl_module):
        """Check if target pruning level reached."""
        if self.current_pruning_total >= self.target_pruning_level:
            print(f"\n✓ Target pruning level {self.target_pruning_level*100:.1f}% reached!")
            print(f"  Actual: {self.current_pruning_total*100:.2f}%")
            print("  Stopping training to finalize pruning.")
            trainer.should_stop = True
    
    def on_train_end(self, trainer, pl_module):
        """Finalize pruning by making zeros permanent - ONLY at the very end."""
        layer = pl_module.net.hidden_layer[self.layer_name]
        if hasattr(layer, 'weight_mask'):
            print("\nFinalizing pruning (making zeros permanent)...")
            
            # Remove mask to make zeros permanent
            prune.remove(layer, 'weight')
            
            # Quick final stats
            weights = layer.weight.data
            zero_count = (weights == 0).sum().item()
            total_count = weights.numel()
            final_pruning = zero_count / total_count
            
            print(f"Final pruning: {final_pruning*100:.2f}% ({zero_count:,}/{total_count:,} weights)")


def compress_model(model, train_loader, val_loader, pruning_level, bit_width, logger, 
                                       device, pruning_step_size=0.05):
    """
    Compress model using a single trainer with iterative pruning via callback.
    
    Args:
        model: The model to compress
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        pruning_level: Target total pruning level (e.g., 0.9 for 90%)
        bit_width: Bit width for quantization
        device: GPU device index
        pruning_step_size: Absolute increase in pruning per step
        total_epochs: Total epochs to train (should be >= steps needed)
    """
    print(f"Starting compression: {pruning_level*100:.1f}% iterative pruning, {bit_width}-bit quantization")
    
    # Check initial state
    print("Initial state:")
    analyze_compression(model)
    
    # Calculate how to distribute pruning over epochs
    n_pruning_steps = int(pruning_level / pruning_step_size)
    epochs_per_step = 1
    
    print(f"\nPruning schedule:")
    print(f"  Total pruning steps: {n_pruning_steps}")
    print(f"  Epochs per pruning step: {epochs_per_step}")
    print(f"  Pruning every {epochs_per_step} epoch(s)")
    
    # Create the pruning callback
    pruning_callback = IterativePruningCallback(
        target_pruning_level=pruning_level,
        pruning_step_size=pruning_step_size,
        start_epoch=0,
        prune_every_n_epochs=epochs_per_step,
        layer_name="out"
    )
    
    # Create single trainer with the pruning callback
    trainer = Trainer(
        accelerator="gpu",
        devices=[device],
        max_epochs=10000000,  # Large number, pruning callback will stop training
        callbacks=[pruning_callback],
        enable_checkpointing=False,
        enable_progress_bar=True,
        logger=logger,
        log_every_n_steps=1,
    )
    
    # Train with iterative pruning
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # Apply quantization after pruning is complete
    if bit_width < 16:
        print(f"\nQuantizing weights to {bit_width} bits...")
        with torch.no_grad():
            layer = model.net.hidden_layer["out"]
            original_weights = layer.weight.data.clone()
            quantized_weights = quantize_weights_manual(original_weights, bit_width)
            layer.weight.data = quantized_weights
    model.net.units["out"].quantize = True
    
    print("\nFinal compression results:")
    analyze_compression(model)
    
    return model
