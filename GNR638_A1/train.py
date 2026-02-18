"""
ULTRA-FAST Training - Target: 2-3 minutes per epoch
Optimized for maximum speed on CPU
"""
import argparse
import json
import time
import pickle
from framework import CrossEntropyLoss, SGD
from framework.utils.parallel_data import ParallelImageDataset, FastDataLoader, split_dataset
from train_model import create_ultra_fast_model

print("\n" + "="*70)
print("⚡ ULTRA-FAST TRAINING MODE ⚡")
print("Target: 2-3 minutes per epoch")
print("="*70)

def accuracy_fast(predictions, labels):
    """Optimized accuracy calculation"""
    batch_size = predictions.shape[0]
    num_classes = predictions.shape[1]
    correct = 0
    
    for b in range(batch_size):
        max_idx = 0
        max_val = predictions.data[b][0]
        for i in range(1, num_classes):
            if predictions.data[b][i] > max_val:
                max_val = predictions.data[b][i]
                max_idx = i
        if max_idx == labels[b]:
            correct += 1
    
    return correct / batch_size


def train_epoch_fast(model, train_loader, criterion, optimizer, epoch, print_every=20):
    """Ultra-fast training - minimal overhead"""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    epoch_start = time.time()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics (minimal computation)
        total_loss += loss.data
        if batch_idx % print_every == 0:  # Only compute acc occasionally
            acc = accuracy_fast(outputs, labels)
            total_acc += acc * print_every
        
        num_batches += 1
        
        # Print less frequently to save time
        if batch_idx % print_every == 0:
            elapsed = time.time() - epoch_start
            batches_done = batch_idx + 1
            time_per_batch = elapsed / batches_done if batches_done > 0 else 0
            eta = time_per_batch * (len(train_loader) - batches_done)
            
            print(f"Epoch {epoch} | Batch {batch_idx+1:3d}/{len(train_loader):3d} | "
                  f"Loss: {loss.data:6.4f} | ETA: {eta:.0f}s", end='\r')
    
    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / num_batches
    
    print(f"\nEpoch {epoch}: Loss={avg_loss:.4f}, Time={epoch_time/60:.2f} min")
    
    return avg_loss, epoch_time


def validate_fast(model, val_loader, criterion):
    """Fast validation - only at end"""
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    
    for images, labels in val_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.data
        
        # Count correct
        batch_size = outputs.shape[0]
        for b in range(batch_size):
            max_idx = 0
            max_val = outputs.data[b][0]
            for i in range(1, outputs.shape[1]):
                if outputs.data[b][i] > max_val:
                    max_val = outputs.data[b][i]
                    max_idx = i
            if max_idx == labels[b]:
                total_correct += 1
            total_samples += 1
    
    acc = total_correct / total_samples
    loss = total_loss / len(val_loader)
    return loss, acc


def main():
    parser = argparse.ArgumentParser(description='Ultra-Fast Training')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='model_ultrafast.pkl')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)  # Large batch!
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--model_type', type=str, default='ultra', 
                       choices=['ultra', 'tiny'],
                       help='ultra=single conv, tiny=no conv (fastest)')
    parser.add_argument('--validate_every', type=int, default=5,
                       help='Validate every N epochs (0=only at end)')
    
    args = parser.parse_args()
    
    print("\nConfiguration:")
    print(f"  Data: {args.data_path}")
    print(f"  Model: {args.model_type}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Train split: {args.train_split}")
    print("="*70)
    
    # Load data with fewer workers (reduce overhead)
    print("\n📂 Loading dataset...")
    dataset = ParallelImageDataset(
        args.data_path,
        target_size=(32, 32),
        num_workers=8  # Fewer workers, less overhead
    )
    
    # Split
    train_dataset, val_dataset = split_dataset(dataset, args.train_split, seed=42)
    
    # Large batch data loaders
    train_loader = FastDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = FastDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create ultra-fast model
    print("\n🚀 Creating model...")
    num_classes = len(dataset.classes)
    model = create_ultra_fast_model(num_classes, model_type=args.model_type)
    
    # Setup training
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    # Training loop
    print("\n" + "="*70)
    print("🏃 STARTING TRAINING")
    print("="*70)
    
    train_start = time.time()
    history = {'train_loss': [], 'epoch_times': []}
    
    for epoch in range(1, args.epochs + 1):
        # Train
        loss, epoch_time = train_epoch_fast(
            model, train_loader, criterion, optimizer, epoch, 
            print_every=20  # Print every 20 batches
        )
        
        history['train_loss'].append(loss)
        history['epoch_times'].append(epoch_time)
        
        # Validate occasionally
        if args.validate_every > 0 and epoch % args.validate_every == 0:
            val_loss, val_acc = validate_fast(model, val_loader, criterion)
            print(f"  Validation: Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        
        # Progress
        total_elapsed = time.time() - train_start
        avg_epoch_time = total_elapsed / epoch
        remaining = avg_epoch_time * (args.epochs - epoch)
        print(f"  Progress: {epoch}/{args.epochs} | "
              f"Avg: {avg_epoch_time/60:.2f} min/epoch | "
              f"ETA: {remaining/60:.1f} min\n")
    
    # Final validation
    print("\n📊 Final Validation...")
    val_loss, val_acc = validate_fast(model, val_loader, criterion)
    
    total_time = time.time() - train_start
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Average per epoch: {total_time/args.epochs/60:.2f} minutes")
    print(f"Final validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    
    # Check if we met the target
    avg_epoch_time = total_time / args.epochs
    
    # Save model
    print(f"\n💾 Saving model to {args.save_path}")
    state = {name: param.data for name, param in model.named_parameters()}
    with open(args.save_path, 'wb') as f:
        pickle.dump(state, f)
    
    # Save history
    history_path = args.save_path.replace('.pkl', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    


if __name__ == '__main__':
    main()
