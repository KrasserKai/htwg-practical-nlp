import torch


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Perform one epoch of training."""
    model.train()
    total_train_loss = 0.0

    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())

        # Backward pass
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    return avg_train_loss


def evaluate_model(model, val_loader, criterion, device):
    """Evaluate the model on the validation set."""
    model.eval()
    total_val_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            total_val_loss += loss.item()

            # Compute accuracy
            predictions = (outputs.squeeze() >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = correct / total
    return avg_val_loss, val_accuracy


def save_best_model(model, val_loss, best_val_loss, patience_counter, model_output):
    """Check if the current validation loss is the best and save the model."""
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), model_output)
        print("  Validation loss improved. Model saved!")
        return val_loss, 0  # Reset patience counter
    else:
        patience_counter += 1
        print(
            f"  No improvement in validation loss. Patience counter: {patience_counter}"
        )
        return best_val_loss, patience_counter


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    patience,
    early_stopping,
    model_output,
):
    """Train the model with early stopping and track training history."""
    model.to(device)
    best_val_loss = float("inf")
    patience_counter = 0

    # History lists to track metrics for later visualization
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")

        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        print(f"  Train Loss: {train_loss:.4f}")

        # Evaluate on validation set
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Acc:    {val_accuracy:.4f}")

        # Check for early stopping
        best_val_loss, patience_counter = save_best_model(
            model, val_loss, best_val_loss, patience_counter, model_output
        )
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

    # Load the best model after training
    model.load_state_dict(torch.load(model_output, weights_only=True))
    print("Best model loaded!")

    # Return training history for visualization
    return train_losses, val_losses, val_accuracies
