from dataloader import get_dataloaders

def main():
    batch_size = 8
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    print("✅ DataLoaders created successfully!\n")

    total_batches = 0
    for i, (inputs, targets) in enumerate(train_loader):
        total_batches += 1
        print(f"Batch {i+1}")
        print(f"Input batch shape: {inputs.shape}")   # [B, 2, 256, 256]
        print(f"Target batch shape: {targets.shape}") # [B, 1, 256, 256]\n")

    print(f"Total training batches: {total_batches}")
    print(f"Total training samples: {len(train_loader.dataset)}")
    print(f"Batch size: {batch_size}")

if __name__ == "__main__":
    main()
